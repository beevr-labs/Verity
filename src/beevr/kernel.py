"""Governance Kernel — doc 15, FR-AG-01/02/03/04/05/06.

Contract: `LLM proposes -> Kernel validates -> (HITL if consequential) ->
execute (idempotent) -> audit`. The model never executes a tool directly.
Kernel checks: action whitelist, per-run/per-action budget, kill-switch state,
role, and whether the action is consequential (=> HITL checkpoint required).

Release-gating: TC-301 (no ungoverned execution), TC-308 (kill switch leaves no
partial side-effects). Also TC-302 (over-budget blocked).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


class KernelReject(Exception):
    def __init__(self, code: str, reason: str = ""):
        self.code = code
        super().__init__(f"{code}: {reason}" if reason else code)


@dataclass(frozen=True)
class ActionSpec:
    name: str
    consequential: bool = False
    checkpoint: bool = False           # requires HITL when consequential
    allowed_roles: tuple[str, ...] = ()  # empty => any role


@dataclass(frozen=True)
class Budget:
    max_tool_calls: int = 50
    max_tokens: int = 400_000
    max_action_tokens: int = 40_000


@dataclass(frozen=True)
class Policy:
    actions: dict[str, ActionSpec]
    budget: Budget = field(default_factory=Budget)
    default_consequential_checkpoint: bool = True

    @classmethod
    def from_config(cls, cfg: dict) -> "Policy":
        specs = {}
        for a in cfg.get("actions", []):
            specs[a["name"]] = ActionSpec(
                name=a["name"],
                consequential=a.get("consequential", False),
                checkpoint=a.get("checkpoint", False),
                allowed_roles=tuple(a.get("allowed_roles", ())),
            )
        b = cfg.get("budget", {})
        return cls(actions=specs, budget=Budget(**b) if b else Budget(),
                   default_consequential_checkpoint=cfg.get(
                       "default_consequential_checkpoint", True))


@dataclass
class ActionProposal:
    action_type: str
    args: dict
    est_tokens: int = 0
    idempotency_key: str | None = None


@dataclass(frozen=True)
class Decision:
    approved: bool
    code: str                 # "OK" | "KERNEL_REJECTED" | "BUDGET_EXCEEDED" | ...
    needs_checkpoint: bool = False
    reason: str = ""


class Kernel:
    def __init__(self, policy: Policy):
        self.policy = policy
        self._tool_calls = 0
        self._tokens = 0
        self._killed_run = False
        self._killed_global = False
        self._executed: dict[str, object] = {}   # idempotency_key -> result

    # --- kill switch (FR-AG-06) ---
    def kill_run(self) -> None:
        self._killed_run = True

    def kill_all(self) -> None:
        self._killed_global = True

    @property
    def killed(self) -> bool:
        return self._killed_run or self._killed_global

    # --- validation (FR-AG-01/02/03) ---
    def validate(self, proposal: ActionProposal, *, role: str = "user") -> Decision:
        if self.killed:
            return Decision(False, "KILL_SWITCH_ACTIVE")
        spec = self.policy.actions.get(proposal.action_type)
        if spec is None:                                   # FR-AG-02 / TC-301
            return Decision(False, "KERNEL_REJECTED",
                            reason=f"action {proposal.action_type!r} not in whitelist")
        if spec.allowed_roles and role not in spec.allowed_roles:
            return Decision(False, "FORBIDDEN_ROLE",
                            reason=f"role {role!r} not permitted for {spec.name}")
        if proposal.est_tokens > self.policy.budget.max_action_tokens:   # FR-AG-03
            return Decision(False, "BUDGET_EXCEEDED", reason="per-action token cap")
        if self._tool_calls + 1 > self.policy.budget.max_tool_calls:
            return Decision(False, "BUDGET_EXCEEDED", reason="per-run tool-call cap")
        if self._tokens + proposal.est_tokens > self.policy.budget.max_tokens:
            return Decision(False, "BUDGET_EXCEEDED", reason="per-run token cap")

        needs = spec.consequential and (
            spec.checkpoint or self.policy.default_consequential_checkpoint)
        return Decision(True, "OK", needs_checkpoint=needs)

    # --- execution (FR-AG-04/05/06) ---
    def execute(self, proposal: ActionProposal, decision: Decision,
                effect: Callable[[], object], *, approved: bool = False) -> object:
        """Execute exactly once. Refuses if the Kernel didn't approve, if a
        required checkpoint wasn't human-approved, or if the kill switch is set
        (=> no partial side-effects, TC-308)."""
        if not decision.approved:
            raise KernelReject(decision.code, decision.reason)
        if decision.needs_checkpoint and not approved:      # FR-AG-04
            raise KernelReject("CHECKPOINT_REQUIRED", "HITL approval needed")
        if self.killed:                                     # FR-AG-06 / TC-308
            raise KernelReject("KILL_SWITCH_ACTIVE", "halted before side-effect")

        key = proposal.idempotency_key
        if key is not None and key in self._executed:       # FR-AG-05 exactly-once
            return self._executed[key]

        result = effect()                                   # the only side-effect
        self._tool_calls += 1
        self._tokens += proposal.est_tokens
        if key is not None:
            self._executed[key] = result
        return result
