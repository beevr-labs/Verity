"""REST API server — doc 11 (Internal API Contract). FastAPI.

Exposes the reference matter-Q&A workflow over the doc 11 contract, wiring the
offline cores: Store (isolation), IngestQueue (ingestion), pipeline.matter_qa
(retrieve→fuse→verify→abstain), AuditLog, and the EgressGuard (default-deny).

Auth is a stub SSO: `POST /auth/token` stands in for the SAML/OIDC callback and
mints an HMAC bearer token carrying {sub, roles, matter_grants} (doc 11 §7).
Production validates a real IdP assertion; the RBAC/session plumbing is identical.

All traffic is in-boundary; the app has no outbound client except the (disabled)
EgressGuard escalation path (doc 16 §1).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass, field

from fastapi import Body, FastAPI, Header, Request
from fastapi.responses import JSONResponse

from .audit import AuditLog
from .kernel import KernelReject
from .queue import IngestQueue
from .store import AccessDenied, Session, Store
from .verification import NLI, lexical_overlap


# --------------------------------------------------------------------------
# App state (single-tenant; one deployment == one instance)
# --------------------------------------------------------------------------
@dataclass
class AppState:
    secret: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    store: Store = None            # type: ignore
    audit: AuditLog = None         # type: ignore
    queue: IngestQueue = field(default_factory=IngestQueue)
    nli: NLI = staticmethod(lambda span, claim: lexical_overlap(claim, span))
    extractor: object | None = None    # None -> agent uses RuleExtractor
    llm: object | None = None          # None -> Q&A runs assertion-mode only
    oidc: object | None = None         # OidcConfig -> enables /auth/oidc
    model_mode: str = "stub"           # "real" when doc-14 roster is loaded
    idempotency: dict = field(default_factory=dict)
    agent_runs: dict = field(default_factory=dict)
    killed_all: bool = False           # global kill switch (FR-AG-06)
    _ids: int = 0

    def next_id(self, prefix: str) -> str:
        self._ids += 1
        return f"{prefix}-{self._ids:06d}"

    @classmethod
    def with_real_models(cls, device: str | None = None) -> "AppState":
        """Production state: bge-m3 + DeBERTa NLI (doc 14). Falls back to the
        lexical stub NLI if the model runtime isn't installed."""
        state = cls()
        try:
            from .models import load_default_runtime
            rt = load_default_runtime(device=device)
            state.nli = rt["nli"]
            state.embedder = rt["embedder"]
            state.router = rt["router"]
            state.model_mode = "real"
        except ImportError:
            pass  # stubs remain; /readyz reports degraded mode
        try:
            from .models import BgeReranker
            state.reranker = BgeReranker()          # CPU by default (see models.py)
        except Exception:
            pass  # optional; compose falls back to plain RRF order
        try:
            from .llm import LlmExtractor, TransformersLLM
            # dev-box tier: one resident generative model; 3B reads legal
            # excerpts reliably where 1.5B refuses (see PROGRESS stress test).
            # Pilot 48GB runs the doc-14 fast/strong split via vLLM instead.
            llm = TransformersLLM("Qwen/Qwen2.5-3B-Instruct", device=device)
            state.llm = llm                      # generative Q&A (compose.py)
            state.extractor = LlmExtractor(llm)
            state.model_mode = "real"
        except Exception:
            pass  # RuleExtractor remains; Q&A stays assertion-mode
        return state


# --------------------------------------------------------------------------
# Stub SSO token (HMAC-signed). Production: validate SAML/OIDC assertion.
# --------------------------------------------------------------------------
def mint_token(state: AppState, sub: str, roles: list[str],
               matter_grants: list[str], walled: list[str] | None = None) -> str:
    body = json.dumps({"sub": sub, "roles": roles, "grants": matter_grants,
                       "walled": walled or []}, separators=(",", ":")).encode()
    b = base64.urlsafe_b64encode(body).rstrip(b"=")
    sig = hmac.new(state.secret, b, hashlib.sha256).digest()
    s = base64.urlsafe_b64encode(sig).rstrip(b"=")
    return f"{b.decode()}.{s.decode()}"


class AuthError(Exception):
    pass


def _session_from_token(state: AppState, token: str) -> Session:
    try:
        b, s = token.encode().split(b".")
    except ValueError:
        raise AuthError("malformed token")
    expected = base64.urlsafe_b64encode(
        hmac.new(state.secret, b, hashlib.sha256).digest()).rstrip(b"=")
    if not hmac.compare_digest(s, expected):
        raise AuthError("bad signature")
    claims = json.loads(base64.urlsafe_b64decode(b + b"=="))
    return Session(user_id=claims["sub"], role=(claims["roles"] or ["user"])[0],
                   matter_grants=frozenset(claims["grants"]),
                   walled_groups=frozenset(claims.get("walled", [])))


def create_real_app() -> FastAPI:
    """uvicorn factory for real-models mode (bge-m3 + NLI + Qwen on GPU):
    `uvicorn beevr.api:create_real_app --factory`. Loads weights at startup."""
    return create_app(AppState.with_real_models())


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------
def create_app(state: AppState | None = None) -> FastAPI:
    if state is None:
        state = AppState()
    if state.audit is None:
        state.audit = AuditLog()
    if state.store is None:
        state.store = Store(audit=state.audit)

    app = FastAPI(title="BeevR for Legal API", version="v1")
    app.state.beevr = state

    def _err(code: str, message: str, status: int, request: Request) -> JSONResponse:
        rid = request.headers.get("x-request-id", "-")
        return JSONResponse(status_code=status, content={
            "error": {"code": code, "message": message, "request_id": rid, "details": {}}})

    @app.exception_handler(AuthError)
    async def _auth(request: Request, exc: AuthError):
        return _err("UNAUTHENTICATED", str(exc), 401, request)

    @app.exception_handler(AccessDenied)
    async def _denied(request: Request, exc: AccessDenied):
        return _err("FORBIDDEN_MATTER", str(exc), 403, request)

    @app.exception_handler(KernelReject)
    async def _kernel(request: Request, exc: KernelReject):
        return _err(exc.code, str(exc), 409, request)

    def require_session(authorization: str | None) -> Session:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise AuthError("missing bearer token")
        return _session_from_token(state, authorization.split(" ", 1)[1])

    # ---- client SPA (doc 17; in-boundary, no external assets) ----
    @app.get("/")
    def index():
        from pathlib import Path

        from fastapi.responses import HTMLResponse
        html = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    # ---- health (no customer data) ----
    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        # degraded CPU-only/stub mode is reported, not hidden (SRS §3.1.2)
        return {"status": "ready", "models": state.model_mode}

    # ---- stub SSO (demo) ----
    @app.post("/auth/token")
    def auth_token(payload: dict = Body(...)):
        tok = mint_token(state, payload["sub"], payload.get("roles", ["user"]),
                         payload.get("matter_grants", []), payload.get("walled"))
        return {"token": tok}

    # ---- real SSO: OIDC id_token -> session token (doc 11 §7, FR-SE-03) ----
    @app.post("/auth/oidc")
    def auth_oidc(payload: dict = Body(...)):
        from .sso import SsoError, validate_id_token
        if state.oidc is None:
            return JSONResponse(status_code=501, content={"error": {
                "code": "VALIDATION", "message": "OIDC not configured",
                "request_id": "-", "details": {}}})
        try:
            s = validate_id_token(state.oidc, payload["id_token"])
        except SsoError as ex:
            return JSONResponse(status_code=401, content={"error": {
                "code": "UNAUTHENTICATED", "message": str(ex),
                "request_id": "-", "details": {}}})
        tok = mint_token(state, s.user_id, [s.role], sorted(s.matter_grants),
                         sorted(s.walled_groups))
        return {"token": tok}

    _register_resource_routes(app, state, require_session)
    return app


def _register_resource_routes(app, state: AppState, require_session):
    from .pipeline import matter_qa

    @app.post("/matters")
    def create_matter(payload: dict = Body(...),
                      authorization: str = Header(None)):
        require_session(authorization)  # any authenticated user; grants gate access
        mid = payload.get("id") or state.next_id("matter")
        state.store.put_matter(mid, client=payload["client"], name=payload["name"],
                               ethical_wall_group=payload.get("ethical_wall_group"))
        return {"id": mid}

    @app.get("/matters")
    def list_matters(authorization: str = Header(None)):
        s = require_session(authorization)
        return {"matters": state.store.list_matters(s)}

    @app.post("/matters/{matter_id}/documents")
    def upload_document(matter_id: str, payload: dict = Body(...),
                        authorization: str = Header(None),
                        idempotency_key: str = Header(None)):
        """Upload with optional inline content. `content` (utf-8 text) or
        `content_b64` triggers REAL ingestion (parse->chunk->embed->store);
        without content the document is queued only (doc 11 §3.2)."""
        import base64 as _b64

        from .ingest import ingest_document

        s = require_session(authorization)
        state.store._authorize(s, matter_id, "upload_document")  # isolation gate
        if idempotency_key and idempotency_key in state.idempotency:
            return state.idempotency[idempotency_key]
        did = state.next_id("doc")
        filename = payload.get("filename", "upload.pdf")
        state.store.put_document(did, matter_id, mime=payload.get("mime", "application/pdf"),
                                 filename=filename)
        data: bytes | None = None
        if "content_b64" in payload:
            data = _b64.b64decode(payload["content_b64"])
        elif "content" in payload:
            data = payload["content"].encode("utf-8")

        if data is not None:
            report = ingest_document(state.store, matter_id, did, data, filename,
                                     embedder=getattr(state, "embedder", None))
            if report.status == "failed":
                state.store.documents[did].data["ingest_status"] = "failed"
                return JSONResponse(status_code=422, content={"error": {
                    "code": "UNSUPPORTED_TYPE", "message": report.error or "",
                    "request_id": "-", "details": {}}})
            state.store.documents[did].data["ingest_status"] = "done"
            resp = {"document_id": did, "status": "done", "chunks": report.chunks}
        else:
            state.queue.enqueue(did, now=0.0)
            resp = {"document_id": did, "status": "queued"}
        if idempotency_key:
            state.idempotency[idempotency_key] = resp
        return resp

    @app.get("/matters/{matter_id}/documents")
    def list_documents(matter_id: str, authorization: str = Header(None)):
        s = require_session(authorization)
        docs = state.store.list_documents(s, matter_id)
        return {"documents": [
            {"document_id": d.id,
             "ingest_status": d.data.get("ingest_status")
                              or state.queue.document_status(d.id)}
            for d in docs]}

    @app.post("/matters/{matter_id}/queries")
    def query(matter_id: str, payload: dict = Body(...),
              authorization: str = Header(None), idempotency_key: str = Header(None)):
        s = require_session(authorization)
        if idempotency_key and idempotency_key in state.idempotency:
            return state.idempotency[idempotency_key]
        embedder = getattr(state, "embedder", None)
        answer_text = None
        if state.llm is not None:                       # generative Q&A (compose.py)
            from .compose import matter_ask
            result = matter_ask(state.store, s, matter_id, payload["question"],
                                llm=state.llm, nli=state.nli, embedder=embedder,
                                reranker=getattr(state, "reranker", None),
                                ts=payload.get("ts", ""))
            answer_text = result.answer_text
        else:                                           # assertion-check fallback
            result = matter_qa(state.store, s, matter_id, payload["question"],
                               nli=state.nli, embedder=embedder,
                               ts=payload.get("ts", ""))
        answer_id = state.next_id("ans")
        resp = {
            "answer_id": answer_id,
            "abstained": result.abstained,
            "confidence": result.confidence,
            "answer_text": answer_text,
            "claims": [
                {"text": c.text, "citations": [
                    {"citation_id": state.next_id("cite"),
                     "document_id": cit.locator.document_id,
                     "locator": {"type": cit.locator.kind, "page": cit.locator.page,
                                 "char_range": list(cit.locator.char_range)},
                     "verified": cit.verified, "snippet": cit.snippet}
                    for cit in c.citations]}
                for c in result.claims],
        }
        if idempotency_key:
            state.idempotency[idempotency_key] = resp
        return resp

    @app.post("/matters/{matter_id}/playbook-review")
    def playbook_review(matter_id: str, payload: dict = Body(default={}),
                        authorization: str = Header(None)):
        """Contract review vs playbook (US-303b). Flags cite real clauses
        (TC-306); suggested redlines are proposals, never applied (TC-307)."""
        from .playbook import FI_DEFAULT_PLAYBOOK, Rule, review
        s = require_session(authorization)
        if state.llm is None:
            return JSONResponse(status_code=501, content={"error": {
                "code": "VALIDATION", "message": "playbook review requires the "
                "model runtime (real-models mode)", "request_id": "-", "details": {}}})
        rules = None
        if payload.get("rules"):
            rules = [Rule(r["id"], r.get("topic", ""), r["standard_position"],
                          r.get("severity", "medium"), r.get("keywords", ""))
                     for r in payload["rules"]]
        flags = review(state.store, s, matter_id, llm=state.llm, rules=rules,
                       embedder=getattr(state, "embedder", None),
                       ts=payload.get("ts", ""))
        return {"flags": [{
            "rule_id": f.rule_id, "topic": f.topic, "severity": f.severity,
            "status": f.status, "quote": f.quote, "rationale": f.rationale,
            "chunk_id": f.chunk_id, "suggested_redline": f.suggested_redline,
        } for f in flags]}

    @app.get("/matters/{matter_id}/briefing")
    def briefing(matter_id: str, authorization: str = Header(None),
                 today: str | None = None):
        """Proactive brief: deadline radar + action items + suggested questions.
        `today` (YYYY-MM-DD) is injectable for tests; defaults to the real date."""
        from datetime import date as _date

        from .briefing import build_briefing
        s = require_session(authorization)
        t = _date.fromisoformat(today) if today else _date.today()
        b = build_briefing(state.store, s, matter_id, today=t,
                           agent_runs=state.agent_runs)
        return {
            "deadlines": [{
                "due": d.due.isoformat(), "days_left": d.days_left,
                "status": d.status, "context": d.context,
                "chunk_id": d.chunk_id, "date_text": d.date_text,
            } for d in b.deadlines],
            "actions": b.actions,
            "suggested_questions": b.suggested_questions,
        }

    @app.get("/matters/{matter_id}/source/{chunk_id}")
    def source(matter_id: str, chunk_id: str, authorization: str = Header(None)):
        """Click-through source viewer (S5, FR-UX-01): the full chunk text so the
        client can highlight the cited span. RBAC + isolation enforced."""
        s = require_session(authorization)
        state.store._authorize(s, matter_id, "read_source")
        row = state.store.chunks.get(chunk_id)
        if row is None or row.matter_id != matter_id:
            return JSONResponse(status_code=404, content={"error": {
                "code": "NOT_FOUND", "message": "chunk not found in matter",
                "request_id": "-", "details": {}}})
        return {"chunk_id": chunk_id, "document_id": row.data.get("document_id"),
                "text": row.data["text"]}

    # ---- governed agent workflow (doc 11 §3.4) ----
    @app.post("/matters/{matter_id}/agent/runs")
    def start_agent_run(matter_id: str, payload: dict = Body(...),
                        authorization: str = Header(None)):
        from .agent import AgentRun
        s = require_session(authorization)
        state.store._authorize(s, matter_id, "agent_run")
        workflow = payload.get("workflow", "obligation_extraction")
        if workflow != "obligation_extraction":
            return JSONResponse(status_code=422, content={"error": {
                "code": "VALIDATION", "message": f"unknown workflow {workflow!r}",
                "request_id": "-", "details": {}}})
        rid = state.next_id("run")
        kwargs = {"extractor": state.extractor} if state.extractor else {}
        run = AgentRun(run_id=rid, matter_id=matter_id, session=s,
                       store=state.store, nli=state.nli, **kwargs)
        if state.killed_all:
            run.kernel.kill_all()
        run.extract_phase(payload.get("input", {}).get("document_ids") or None,
                          ts=payload.get("ts", ""))
        state.agent_runs[rid] = run
        return {"run_id": rid, "status": run.status,
                "proposals": _proposals(run)}

    @app.get("/agent/runs/{run_id}")
    def get_agent_run(run_id: str, authorization: str = Header(None)):
        s = require_session(authorization)
        run = state.agent_runs[run_id]
        state.store._authorize(s, run.matter_id, "agent_run_read")
        return {"run_id": run_id, "status": run.status,
                "proposals": _proposals(run),
                "persisted": run.persisted}

    @app.post("/agent/proposals/{run_id}/{index}/decision")
    def decide_proposal(run_id: str, index: int, payload: dict = Body(...),
                        authorization: str = Header(None),
                        idempotency_key: str = Header(None)):
        s = require_session(authorization)
        run = state.agent_runs[run_id]
        state.store._authorize(s, run.matter_id, "agent_decision")
        item = run.decide(index, payload["decision"],
                          payload.get("edited_args", {}).get("text"),
                          idempotency_key=idempotency_key,
                          ts=payload.get("ts", ""))
        return {"status": item.status}

    @app.post("/agent/runs/{run_id}/kill")
    def kill_run(run_id: str, authorization: str = Header(None)):
        s = require_session(authorization)
        run = state.agent_runs[run_id]
        state.store._authorize(s, run.matter_id, "agent_kill")
        run.kill()
        return {"run_id": run_id, "status": run.status}

    @app.post("/admin/agents/kill-all")
    def kill_all(authorization: str = Header(None)):
        s = require_session(authorization)
        if s.role not in ("legal_admin", "counsel_admin"):
            return _err_payload("FORBIDDEN", "admin role required", 403)
        state.killed_all = True
        for run in state.agent_runs.values():
            run.kernel.kill_all()
            run.status = "killed"
        return {"status": "killed_all"}

    def _err_payload(code: str, message: str, status: int):
        return JSONResponse(status_code=status, content={"error": {
            "code": code, "message": message, "request_id": "-", "details": {}}})

    def _proposals(run):
        return [{"index": i, "item_type": it.item_type, "text": it.text,
                 "party": it.party, "trigger_or_due": it.trigger_or_due,
                 "verified": it.verified, "status": it.status,
                 "citation": {"document_id": it.locator.document_id
                              if it.locator else None}}
                for i, it in enumerate(run.items)]

    @app.get("/matters/{matter_id}/audit")
    def audit(matter_id: str, authorization: str = Header(None)):
        s = require_session(authorization)
        state.store._authorize(s, matter_id, "read_audit")
        events = [e for e in state.audit.events if e.target == matter_id]
        return {"events": [
            {"seq": e.seq, "actor": e.actor, "type": e.type, "target": e.target,
             "ts": e.ts, "this_hash": e.this_hash} for e in events]}
