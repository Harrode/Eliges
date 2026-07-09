"""Tier 3 — LLM-Judge (EligES v2 confidence-cascade).

Given a single eligibility criterion and a patient's relevant evidence, an LLM
decides whether the patient MEETS the criterion. This is the generalizable
component: it relies on no dataset-specific rules, so it works on any criterion
or corpus.

Design notes:
- Evidence is extracted (not the whole note) to bound token cost.
- Output is forced to a tiny JSON object; reasoning models are run with
  thinking disabled and the response is stripped/repaired before parsing.
- The judge is deliberately dataset-agnostic; the only domain prior baked into
  the prompt is the *general* clinical-trial convention that ability/attribute
  criteria (e.g., "must speak English") are met by default unless the evidence
  says otherwise.
"""
from __future__ import annotations

import json
import re
import time
from typing import Dict, List, Optional

_STOPWORDS = {
    "the", "a", "an", "of", "or", "and", "to", "in", "on", "for", "with",
    "must", "their", "own", "any", "such", "as", "past", "current", "use",
    "patient", "patients", "history", "within", "more", "than", "over", "at",
    "least", "value", "values", "is", "are", "be", "has", "have", "no", "not",
}


def _content_terms(criterion: str) -> List[str]:
    """Derive search terms from the criterion text (no dataset-specific lists)."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", criterion.lower())
    terms = [w for w in words if w not in _STOPWORDS]
    # de-dup, keep order
    seen, out = set(), []
    for w in terms:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def extract_evidence(patient_text: str, criterion: str, max_chars: int = 2400) -> str:
    """Select criterion-relevant sentences (+neighbours) up to max_chars.

    Falls back to the head of the note when nothing matches, so the judge always
    receives some context (important for default-met criteria where the relevant
    signal is the *absence* of disqualifying evidence)."""
    terms = _content_terms(criterion)
    sentences = _split_sentences(patient_text)
    if not sentences:
        return patient_text[:max_chars]

    scored = []
    for i, sent in enumerate(sentences):
        low = sent.lower()
        score = sum(1 for t in terms if t in low)
        if score:
            scored.append((i, score))
    scored.sort(key=lambda x: (-x[1], x[0]))

    chosen_idx = set()
    for i, _ in scored:
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(sentences):
                chosen_idx.add(j)

    pieces, total = [], 0
    # Always include note head for context (social history, etc.)
    head = patient_text[:600].strip()
    if head:
        pieces.append(head)
        total += len(head)
    for j in sorted(chosen_idx):
        s = sentences[j]
        if total + len(s) + 1 > max_chars:
            break
        pieces.append(s)
        total += len(s) + 1

    evidence = "\n".join(pieces).strip()
    return evidence[:max_chars] if evidence else patient_text[:max_chars]


_JUDGE_SYSTEM = (
    "You are a clinical trial eligibility judge. Given ONE criterion and a "
    "patient's evidence, decide if the patient MEETS the criterion. "
    "Convention: ability/attribute criteria (e.g., speaks English, can make "
    "own decisions) are MET by default unless the evidence indicates otherwise. "
    "Exclusion criteria are MET only if the excluded item is present. "
    'Output ONLY JSON: {"met": true/false, "reason": "<=12 words"}'
)


def _parse_verdict(text: str) -> Optional[Dict]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if "met" not in obj:
        return None
    return {"met": bool(obj["met"]), "reason": str(obj.get("reason", ""))[:200]}


def llm_judge(criterion: str, evidence: str, llm_client, model: str,
              strip_thinking=None) -> Dict:
    """Return {"met": bool, "reason": str, "source": "llm"|"llm_error"}."""
    # Hard cap so input + max_tokens stays within 8192 context (vLLM validates
    # input_tokens <= context_len - max_tokens). Callers should prefer
    # extract_evidence(); this guards direct full-note passes.
    if len(evidence) > 2400:
        evidence = evidence[:2400]
    user = f"CRITERION: {criterion}\n\nPATIENT EVIDENCE:\n{evidence}"
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
    # NOTE: for the qwen endpoint, top-level {"enable_thinking": False} does NOT
    # suppress reasoning; the chat_template_kwargs form does. We pass both for
    # cross-provider robustness and still strip any residual <think> block.
    _no_think = {"enable_thinking": False,
                 "chat_template_kwargs": {"enable_thinking": False}}
    strict = ('\n\nReturn ONLY the JSON object, no reasoning, no markdown.')
    last_text = ""
    # Up to 4 attempts; transient API/network errors (503, connection) get
    # exponential backoff so a flaky endpoint does not corrupt the run.
    for attempt in range(4):
        msgs = messages if attempt == 0 else [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user + strict},
        ]
        try:
            resp = llm_client.chat.completions.create(
                model=model, messages=msgs, temperature=0,
                # Reasoning-suppressed models still emit internal reasoning
                # tokens that count against max_tokens; 128 truncated the JSON
                # and forced conservative not-met defaults (low recall). 1024
                # leaves room for reasoning + the tiny JSON verdict.
                max_tokens=1024, extra_body=_no_think,
            )
            raw = resp.choices[0].message.content or ""
            last_text = strip_thinking(raw) if strip_thinking else raw
            verdict = _parse_verdict(last_text)
            if verdict:
                verdict["source"] = "llm"
                return verdict
            # parsed but no JSON -> retry quickly with strict instruction
        except Exception as e:  # network / API error (503, connection refused, ...)
            last_text = f"ERROR: {e}"
            time.sleep(min(2 ** attempt, 8))  # 1s, 2s, 4s backoff
    # conservative default: not met (flagged so the eval can re-run)
    return {"met": False, "reason": f"parse_failed:{last_text[:60]}", "source": "llm_error"}
