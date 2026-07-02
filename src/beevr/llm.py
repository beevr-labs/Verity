"""Generative LLM layer — doc 14 §1/§5 + doc 15 §6 prompt discipline.

Providers (pluggable, all in-boundary):
  * TransformersLLM  — local weights via HF transformers on the dev/demo GPU.
                       Dev tier runs a small instruct model; the pilot 48GB
                       reference runs Qwen2.5-7B/32B through the same class or
                       a vLLM server (doc 14 §5).
  * OpenAICompatLLM  — client for an in-boundary vLLM/Ollama endpoint
                       (http://localhost:8000/v1-style). The endpoint URL is
                       config; it is NOT the frontier-escalation path and must
                       point inside the boundary.

LlmExtractor implements the agent's Extractor protocol using the doc 15 §6
discipline: fixed output schema, extract-only (no legal conclusions), and the
source span is provided so the model cannot cite anything else. Whatever the
model returns STILL passes the NLI verification gate in the agent pipeline —
a hallucinated item is dropped there, never shown (FR-SF-01).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from .locator import Locator


class LlmProvider(Protocol):
    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> str: ...


# --------------------------------------------------------------------------
# Local transformers provider (deterministic: greedy decoding)
# --------------------------------------------------------------------------
class TransformersLLM:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        import torch
        messages = [{"role": "user", "content": prompt}]
        encoded = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt")
        ids = encoded["input_ids"] if not torch.is_tensor(encoded) else encoded
        ids = ids.to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    def generate_batch(self, prompts: list[str], *,
                       max_new_tokens: int = 512) -> list[str]:
        """One batched GPU pass instead of N serial calls (map-then-verify Q&A
        makes 6-10 small calls per question; batching cuts wall-clock ~Nx).
        Decoder-only models need LEFT padding for correct generation."""
        import torch
        texts = [self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True, tokenize=False)
                 for p in prompts]
        old_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                                 add_special_tokens=False).to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens,
                    do_sample=False, temperature=None, top_p=None, top_k=None,
                    pad_token_id=self.tokenizer.pad_token_id)
        finally:
            self.tokenizer.padding_side = old_side
        n_in = enc["input_ids"].shape[1]
        return [self.tokenizer.decode(o[n_in:], skip_special_tokens=True)
                for o in out]


class OpenAICompatLLM:
    """Client for an in-boundary vLLM/Ollama server (pilot form factor)."""

    def __init__(self, base_url: str, model: str, api_key: str = "none"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        import httpx
        r = httpx.post(f"{self.base_url}/chat/completions", json={
            "model": self.model, "temperature": 0,
            "max_tokens": max_new_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------
# LLM-backed obligation extractor (doc 15 §6 prompt discipline)
# --------------------------------------------------------------------------
PROMPT = """You are an extraction engine for a legal-compliance tool. \
From the CLAUSE below, extract obligations, covenants, termination/renewal \
triggers, key dates, or regulatory requirements — if any.

Rules (strict):
- Output ONLY a JSON array, no prose. Empty array [] if the clause contains none.
- Each element: {{"item_type": "obligation|covenant|date|termination_trigger|regulatory_clause",
  "text": "<the obligation restated from the clause, verbatim-faithful>",
  "party": "<obligated party or empty>", "counterparty": "<beneficiary or empty>",
  "trigger_or_due": "<date/trigger or empty>"}}
- "text" MUST be supported by the clause word-for-word in substance. Never add \
facts, never conclude legal effect, never invent parties or dates.

CLAUSE:
{clause}

JSON:"""

_JSON_ARRAY = re.compile(r"\[.*\]", re.S)


@dataclass
class LlmExtractor:
    """Drop-in for agent.RuleExtractor. Model output is parsed defensively and
    every item still passes the NLI verification gate downstream."""
    llm: LlmProvider
    max_new_tokens: int = 512

    def extract(self, chunk_text: str, locator: Locator) -> list:
        from .agent import Item
        raw = self.llm.generate(PROMPT.format(clause=chunk_text.strip()),
                                max_new_tokens=self.max_new_tokens)
        m = _JSON_ARRAY.search(raw)
        if not m:
            return []                                    # unparseable -> no items
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        items = []
        allowed = {"obligation", "covenant", "date", "termination_trigger",
                   "regulatory_clause"}
        for el in data if isinstance(data, list) else []:
            if not isinstance(el, dict) or el.get("item_type") not in allowed:
                continue
            text = str(el.get("text", "")).strip()
            if not text:
                continue
            items.append(Item(item_type=el["item_type"], text=text,
                              party=str(el.get("party", ""))[:100],
                              counterparty=str(el.get("counterparty", ""))[:100],
                              trigger_or_due=str(el.get("trigger_or_due", ""))[:50],
                              locator=locator))
        return items
