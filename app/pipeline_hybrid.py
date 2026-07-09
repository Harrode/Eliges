"""EligES Hybrid pipeline: L1 parse → L3 ES → L4 rule_filter → selective L4′ judge."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set

from app.cohort_profile import apply_generic_parse_pipeline, apply_profile_overrides
from app.llm_judge import extract_evidence, llm_judge
from app.routing import needs_semantic_judge


def parse_with_profile(
    query: str,
    profile_id: str,
    criterion_id: Optional[str] = None,
    llm_parse_fn=None,
) -> Dict:
    """L1: LLM parse + cohort profile overrides + generic post-process."""
    parse_fn = llm_parse_fn
    if parse_fn is None:
        import backend
        parse_fn = backend.llm_parse_query

    conds = parse_fn(query, profile_id=profile_id, criterion_id=criterion_id)
    if not isinstance(conds, dict):
        conds = {}
    conds = apply_profile_overrides(conds, query, profile_id, criterion_id)
    conds = apply_generic_parse_pipeline(conds, query)
    return {k: v for k, v in conds.items() if v is not None and v != []}


def _patient_id(hit: Dict) -> str:
    return str(
        hit.get("id")
        or hit.get("patient_id")
        or hit.get("source", {}).get("patient_id")
        or ""
    )


def evaluate_criterion(
    criterion: Dict,
    records: List[Dict],
    profile_id: str,
    *,
    use_judge: bool = True,
    es_search_fn=None,
    rule_filter_fn=None,
    llm_client=None,
    llm_model: str = "",
    strip_thinking=None,
    workers: int = 5,
) -> Dict[str, Any]:
    """Evaluate one n2c2 criterion: ES→K→L4→optional judge. Returns metrics dict."""
    import backend

    es_search = es_search_fn or backend.es_search
    rule_filter = rule_filter_fn or backend.rule_filter
    llm_client = llm_client or backend.llm_client
    llm_model = llm_model or backend.LLM_MODEL
    strip_thinking = strip_thinking or backend._strip_thinking

    query = criterion["query"]
    cid = criterion.get("id")
    conds = parse_with_profile(query, profile_id, criterion_id=cid)

    raw = es_search(conds, query) or {}
    hits = raw.get("hits", [])
    hit_ids: Set[str] = {_patient_id(h) for h in hits if _patient_id(h)}

    filtered = rule_filter(hits, conds) if hits else []
    l4_met_ids = {_patient_id(h) for h in filtered if _patient_id(h)}

    use_judge_path = use_judge and needs_semantic_judge(conds, profile_id, cid, query)
    judge_targets = [
        r for r in records
        if r["tags"].get(cid) in ("met", "not met") and r["patient_id"] in hit_ids
    ] if use_judge_path else []

    judge_results: Dict[str, Dict] = {}

    def _judge_one(rec):
        evidence = extract_evidence(rec["text"], query)
        verdict = llm_judge(
            query, evidence, llm_client, llm_model, strip_thinking=strip_thinking,
        )
        return rec["patient_id"], verdict

    if judge_targets:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_judge_one, r) for r in judge_targets]
            for fut in as_completed(futs):
                pid, verdict = fut.result()
                judge_results[pid] = verdict

    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    sources = {"rule": 0, "llm": 0, "llm_error": 0, "es_miss": 0}

    for rec in records:
        label = rec["tags"].get(cid)
        if label not in ("met", "not met"):
            continue
        expected = label == "met"
        pid = rec["patient_id"]

        if pid not in hit_ids:
            predicted = False
            sources["es_miss"] += 1
            src = "es_miss"
        elif use_judge_path and pid in judge_results:
            v = judge_results[pid]
            predicted = bool(v.get("met"))
            src = v.get("source", "llm")
            sources[src] = sources.get(src, 0) + 1
        else:
            predicted = pid in l4_met_ids
            sources["rule"] += 1
            src = "rule"

        if expected and predicted:
            counts["TP"] += 1
        elif not expected and not predicted:
            counts["TN"] += 1
        elif expected and not predicted:
            counts["FN"] += 1
        else:
            counts["FP"] += 1

    return {
        "criterion": cid,
        "query": query,
        "profile_id": profile_id,
        "nlu_conds": conds,
        "es_total": raw.get("total", len(hits)),
        "es_hits": len(hits),
        "after_l4": len(filtered),
        "use_judge": use_judge_path,
        "judge_calls": len(judge_targets),
        "sources": sources,
        **counts,
    }


def _hit_note_text(hit: Dict) -> str:
    import backend
    return backend._patient_note_text(hit)


def verify_search_hits(
    query: str,
    conds: Dict,
    hits: List[Dict],
    profile_id: str,
    *,
    criterion_id: Optional[str] = None,
    use_judge: bool = True,
    rule_filter_fn=None,
    llm_client=None,
    llm_model: str = "",
    strip_thinking=None,
    workers: int = 5,
) -> Dict[str, Any]:
    """L4 verification for live Search — same routing as eval_hybrid, no gold labels.

    Returns verified hits (MET) with verification_source and updated hit_reason.
    """
    import backend

    rule_filter = rule_filter_fn or backend.rule_filter
    llm_client = llm_client or backend.llm_client
    llm_model = llm_model or backend.LLM_MODEL
    strip_thinking = strip_thinking or backend._strip_thinking

    filtered = rule_filter(hits, conds) if hits else []
    l4_met_ids = {_patient_id(h) for h in filtered if _patient_id(h)}

    use_judge_path = (
        use_judge
        and bool(llm_client)
        and llm_model != "rule-only"
        and needs_semantic_judge(conds, profile_id, criterion_id, query)
    )

    judge_results: Dict[str, Dict] = {}

    def _judge_hit(hit: Dict):
        pid = _patient_id(hit)
        evidence = extract_evidence(_hit_note_text(hit), query)
        verdict = llm_judge(
            query, evidence, llm_client, llm_model, strip_thinking=strip_thinking,
        )
        return pid, verdict

    if use_judge_path and hits:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_judge_hit, h) for h in hits]
            for fut in as_completed(futs):
                pid, verdict = fut.result()
                if pid:
                    judge_results[pid] = verdict

    sources = {"rule": 0, "llm": 0, "llm_error": 0}
    verified: List[Dict] = []

    for h in hits:
        pid = _patient_id(h)
        if not pid:
            continue
        met = False
        source = "rule"
        extra_reason = ""

        if use_judge_path and pid in judge_results:
            v = judge_results[pid]
            met = bool(v.get("met"))
            source = v.get("source", "llm")
            sources[source] = sources.get(source, 0) + 1
            if v.get("reason"):
                tag = "MET" if met else "NOT-MET"
                extra_reason = f"[L4.2 judge] {tag}: {v['reason']}"
        else:
            met = pid in l4_met_ids
            if met:
                sources["rule"] += 1

        if not met:
            continue

        out = dict(h)
        reasons = list(out.get("hit_reason") or [])
        if extra_reason:
            reasons.append(extra_reason)
        out["hit_reason"] = reasons
        out["verification_source"] = source
        verified.append(out)

    return {
        "verified_hits": verified,
        "rule_filtered": filtered,
        "use_judge": use_judge_path,
        "judge_calls": len(judge_results) if use_judge_path else 0,
        "sources": sources,
    }
