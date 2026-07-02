"""Kernel governance tests — FR-AG-01/02/03/04/05/06.

Release-gating: TC-301 (no ungoverned execution), TC-308 (kill switch).
"""
import pytest

from beevr.kernel import ActionProposal, Kernel, KernelReject, Policy

POLICY = Policy.from_config({
    "budget": {"max_tool_calls": 3, "max_tokens": 1000, "max_action_tokens": 400},
    "actions": [
        {"name": "read_documents", "consequential": False},
        {"name": "extract_obligations", "consequential": False},
        {"name": "write_extraction_record", "consequential": True, "checkpoint": True},
        {"name": "send_email", "consequential": True, "checkpoint": True,
         "allowed_roles": ["counsel_admin"]},
    ],
})


def _k() -> Kernel:
    return Kernel(POLICY)


@pytest.mark.release_gating
def test_non_whitelisted_action_blocked_before_execution_TC301():
    k = _k()
    effects = []
    p = ActionProposal("delete_everything", {})
    decision = k.validate(p)
    assert decision.approved is False and decision.code == "KERNEL_REJECTED"
    # execution must refuse -> no side-effect ran
    with pytest.raises(KernelReject):
        k.execute(p, decision, lambda: effects.append("boom"))
    assert effects == []


def test_over_budget_action_blocked_TC302():
    k = _k()
    # per-action token cap = 400
    d = k.validate(ActionProposal("extract_obligations", {}, est_tokens=500))
    assert d.code == "BUDGET_EXCEEDED"


def test_per_run_toolcall_cap_TC302():
    k = _k()
    for _ in range(3):  # cap = 3
        p = ActionProposal("read_documents", {})
        k.execute(p, k.validate(p), lambda: "ok")
    d = k.validate(ActionProposal("read_documents", {}))
    assert d.code == "BUDGET_EXCEEDED"


def test_consequential_requires_hitl_checkpoint_FR_AG_04():
    k = _k()
    p = ActionProposal("write_extraction_record", {"item": "covenant"})
    d = k.validate(p)
    assert d.approved and d.needs_checkpoint
    # without human approval -> refused
    with pytest.raises(KernelReject, match="CHECKPOINT_REQUIRED"):
        k.execute(p, d, lambda: "saved")
    # with approval -> executes once
    assert k.execute(p, d, lambda: "saved", approved=True) == "saved"


def test_idempotent_execute_exactly_once_FR_AG_05():
    k = _k()
    calls = []
    p = ActionProposal("read_documents", {}, idempotency_key="k1")
    d = k.validate(p)
    k.execute(p, d, lambda: calls.append(1) or "r")
    k.execute(p, d, lambda: calls.append(1) or "r")  # same key -> cached
    assert calls == [1]


def test_role_gating_send_email():
    k = _k()
    p = ActionProposal("send_email", {"to": "x"})
    assert k.validate(p, role="user").code == "FORBIDDEN_ROLE"
    assert k.validate(p, role="counsel_admin").approved


@pytest.mark.release_gating
def test_kill_switch_no_partial_side_effects_TC308():
    k = _k()
    effects = []
    p = ActionProposal("read_documents", {})
    d = k.validate(p)          # approved before kill
    k.kill_all()               # kill switch flipped mid-run
    with pytest.raises(KernelReject, match="KILL_SWITCH_ACTIVE"):
        k.execute(p, d, lambda: effects.append("side-effect"))
    assert effects == []       # nothing executed
    # further validation also refuses
    assert k.validate(ActionProposal("read_documents", {})).code == "KILL_SWITCH_ACTIVE"
