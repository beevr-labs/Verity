"""Model layer — doc 14: roster, router, real providers.

Roster (doc 14 §1, all local / in-boundary):
  * Embeddings : BAAI/bge-m3 (1024-dim, multilingual)      -> Bgem3Embedder
  * NLI verify : cross-encoder DeBERTa-v3 NLI class        -> CrossEncoderNLI
  * Router     : rule-based fast-vs-strong (doc 14 §2)     -> ModelRouter

Providers lazy-import torch/sentence-transformers so the rest of the codebase
stays importable offline; the router is pure logic (FR-ML-02, TC-903).
Weights are pinned per release; nothing is trained on customer data (CMP-5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Rule-based model router (doc 14 §2, FR-ML-02 -> TC-903)
# --------------------------------------------------------------------------
STRONG_TASKS = frozenset({
    "multi_doc_synthesis", "obligation_extraction", "low_confidence_retry",
})


@dataclass(frozen=True)
class Route:
    model: str            # "fast" | "strong"
    reason: str


@dataclass
class ModelRouter:
    """`retrieval-only / short factual -> fast`; synthesis/extraction/retry -> strong.
    Thresholds are config-driven (Kernel policy, doc 15 §2); every decision is
    traced by the caller."""
    strong_when: frozenset[str] = STRONG_TASKS
    long_question_words: int = 40         # long/multi-part questions -> strong

    def route(self, *, task: str = "qa", question: str = "",
              doc_count: int = 1, retry: bool = False) -> Route:
        if retry:
            return Route("strong", "low_confidence_retry")
        if task in self.strong_when:
            return Route("strong", f"task:{task}")
        if doc_count > 1:
            return Route("strong", "multi_doc_synthesis")
        if len(question.split()) > self.long_question_words:
            return Route("strong", "long_question")
        return Route("fast", "short_factual")


# --------------------------------------------------------------------------
# Real providers (lazy imports; heavy deps only load when used)
# --------------------------------------------------------------------------
class Bgem3Embedder:
    """bge-m3 embeddings (1024-dim) via sentence-transformers. Weights download
    once, then run fully offline (NFR-06)."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class CrossEncoderNLI:
    """NLI cross-encoder for citation verification (doc 13 §2).

    Callable as `nli(premise_span, hypothesis_claim) -> P(entailment) in [0,1]`
    — the exact signature `verification.Verifier` expects. Deterministic
    (no sampling)."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base",
                 device: str | None = None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device)
        # label order for this family: [contradiction, entailment, neutral]
        self.entail_idx = 1

    def __call__(self, premise: str, hypothesis: str) -> float:
        import torch
        logits = self.model.predict([(premise, hypothesis)],
                                    convert_to_numpy=False,
                                    apply_softmax=True)[0]
        if not isinstance(logits, torch.Tensor):
            import torch as _t
            logits = _t.tensor(logits)
        return float(logits[self.entail_idx])


def load_default_runtime(device: str | None = None) -> dict:
    """Load the doc-14 default roster. Raises ImportError if deps missing —
    callers fall back to stubs (and tests skip)."""
    return {
        "embedder": Bgem3Embedder(device=device),
        "nli": CrossEncoderNLI(device=device),
        "router": ModelRouter(),
    }
