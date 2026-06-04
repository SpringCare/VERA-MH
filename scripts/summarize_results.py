#!/usr/bin/env python3
"""
Aggregate judge results.csv into stats JSON and a Markdown improvement report.

Reads per-dimension scores and *_yes_question_id / *_yes_reasoning columns,
joins rubric question text, and emits global + per-dimension failure breakdowns.

Usage:
  uv run python3 scripts/summarize_results.py \\
    --results example.results.csv --rubric data/rubric.tsv \\
    --out-stats output/improvement_stats.json --out-md output/improvement_report.md
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, cast

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from judge.constants import BEST_PRACTICE, DAMAGING, NEUTRAL, NOT_RELEVANT  # noqa: E402
from judge.rubric_config import (  # noqa: E402
    COL_DIMENSION,
    COL_QUESTION,
    COL_QUESTION_ID,
    COL_SEVERITY,
)
from judge.score_utils import DIMENSIONS, calculate_dimension_scores  # noqa: E402

# Order here only affects the first aggregation pass; TL;DR uses HPH then Suboptimal.
FAILURE_OUTCOMES = (DAMAGING, NEUTRAL)
LOW_SAMPLE_THRESHOLD = 30

# Outcome labels align with judge severity mapping (see judge/llm_judge.py).
OUTCOME_BAND_HEADLINE: Dict[str, Tuple[str, str]] = {
    DAMAGING: ("High Potential for Harm", "red"),
    NEUTRAL: ("Suboptimal but Low Potential for Harm", "yellow"),
}


def _parse_question_id(raw: Any) -> Optional[int]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _severity_rank(sev: str) -> int:
    s = (sev or "").strip().lower()
    if s == "red":
        return 2
    if s == "yellow":
        return 1
    return 0


def _outcome_rank(outcome: str) -> int:
    if outcome == DAMAGING:
        return 2
    if outcome == NEUTRAL:
        return 1
    return 0


def load_rubric_question_map(rubric_path: Path) -> Dict[int, Dict[str, str]]:
    """Map Question ID to question text, severity, dimension (longest Question wins)."""
    df = pd.read_csv(rubric_path, sep="\t")
    if COL_QUESTION_ID not in df.columns:
        raise ValueError(f"Rubric missing {COL_QUESTION_ID}: {rubric_path}")

    qid = pd.to_numeric(df[COL_QUESTION_ID], errors="coerce")
    df = df.assign(_qid=qid).copy()
    df["_qid"] = df["_qid"].ffill()
    df = df[df["_qid"].notna()]
    df["_qid"] = df["_qid"].astype(int)

    out: Dict[int, Dict[str, str]] = {}
    qids = sorted(int(x) for x in cast(pd.Series, df["_qid"]).unique())
    for q in qids:
        g = cast(pd.DataFrame, df.loc[df["_qid"] == q])
        texts = cast(pd.Series, g[COL_QUESTION]).dropna().astype(str)
        texts = texts[texts.str.strip() != ""]
        question = max(texts, key=len) if len(texts) else ""
        sev = cast(pd.Series, g[COL_SEVERITY]).dropna().astype(str)
        sev_nonempty = sev[sev.str.strip() != ""].tolist()
        severity = str(sev_nonempty[0]) if sev_nonempty else ""
        dim = cast(pd.Series, g[COL_DIMENSION]).dropna().astype(str)
        dim_nonempty = dim[dim.str.strip() != ""].tolist()
        dimension = str(dim_nonempty[0]) if dim_nonempty else ""
        out[q] = {
            "question": question.strip(),
            "severity": severity.strip(),
            "dimension": dimension.strip(),
        }
    return out


def _collect_usecols(path: Path, dimensions: List[str]) -> frozenset[str]:
    """Column names to load; frozenset for O(1) membership in read_csv."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    want: set[str] = {
        "filename",
        "run_id",
        "persona_name",
        "risk_level",
        "judge_model",
    }
    header_set = set(header)
    for d in dimensions:
        if d not in header_set:
            continue
        want.add(d)
        yid, yreason = f"{d}_yes_question_id", f"{d}_yes_reasoning"
        if yid in header_set:
            want.add(yid)
        if yreason in header_set:
            want.add(yreason)
    return frozenset(want)


def _ordered_dimensions(dimensions: List[str], dims_present: List[str]) -> List[str]:
    """Stable order: canonical list first, then any extra columns from the CSV."""
    present_set = set(dims_present)
    return [d for d in dimensions if d in present_set] + [
        d for d in dims_present if d not in dimensions
    ]


def _global_failure_mode_sort_key(
    item: Tuple[Tuple[str, str, int], int],
    rubric_map: Dict[int, Dict[str, str]],
) -> Tuple[int, int, int, int]:
    (_dim, outcome, qid), cnt = item
    rub = rubric_map.get(qid, {})
    return (
        -cnt,
        -_outcome_rank(outcome),
        -_severity_rank(rub.get("severity", "")),
        qid,
    )


def aggregate_improvements(
    df: pd.DataFrame,
    rubric_map: Dict[int, Dict[str, str]],
    dimensions: List[str],
    *,
    exemplars_per_bucket: int = 3,
    top_questions_per_band: int = 12,
    low_sample_threshold: int = LOW_SAMPLE_THRESHOLD,
) -> Dict[str, Any]:
    total = len(df)
    dims_present = [d for d in dimensions if d in df.columns]
    dim_scores, _ = calculate_dimension_scores(df, detailed=True)

    out: Dict[str, Any] = {
        "meta": {
            "total_conversations": total,
            "dimensions_in_results": dims_present,
            "low_sample": total < low_sample_threshold,
            "low_sample_threshold": low_sample_threshold,
        },
        "dimension_scores": dim_scores,
        "dimensions": {},
    }

    q_counts: DefaultDict[Tuple[str, str, int], int] = defaultdict(int)
    # (filename, judge_model, reasoning) per bucket
    ex_store: DefaultDict[Tuple[str, str, int], List[Tuple[str, str, str]]] = (
        defaultdict(list)
    )
    ex_seen: DefaultDict[Tuple[str, str, int], set] = defaultdict(set)

    for dim in dims_present:
        yid_col = f"{dim}_yes_question_id"
        yreason_col = f"{dim}_yes_reasoning"
        has_yid = yid_col in df.columns
        has_reason = yreason_col in df.columns

        for outcome in FAILURE_OUTCOMES:
            mask = df[dim] == outcome
            sub = df.loc[mask]
            for _, row in sub.iterrows():
                qid = _parse_question_id(row[yid_col]) if has_yid else None
                if qid is None:
                    continue
                key = (dim, outcome, qid)
                q_counts[key] += 1
                if has_reason and exemplars_per_bucket > 0:
                    reason = row[yreason_col]
                    if reason is None or (
                        isinstance(reason, float) and pd.isna(reason)
                    ):
                        continue
                    rs = str(reason).strip()
                    if not rs:
                        continue
                    norm = _norm_text(rs)
                    if norm in ex_seen[key]:
                        continue
                    if len(ex_store[key]) >= exemplars_per_bucket:
                        continue
                    ex_seen[key].add(norm)
                    fn = (
                        str(row.get("filename", "")) if "filename" in df.columns else ""
                    )
                    jm_raw = (
                        row.get("judge_model", "")
                        if "judge_model" in df.columns
                        else ""
                    )
                    if jm_raw is None or (
                        isinstance(jm_raw, float) and pd.isna(jm_raw)
                    ):
                        jm = ""
                    else:
                        jm = str(jm_raw).strip()
                    ex_store[key].append((fn, jm, rs))

    for dim in dims_present:
        dim_entry: Dict[str, Any] = {
            "outcome_counts": {},
            "outcome_pct_of_all": {},
            "outcome_pct_of_relevant": {},
            "by_outcome": {},
        }

        for label in (BEST_PRACTICE, NEUTRAL, DAMAGING, NOT_RELEVANT):
            dim_entry["outcome_counts"][label] = int((df[dim] == label).sum())

        if total > 0:
            dim_entry["outcome_pct_of_all"] = {
                k: round(v / total, 6) for k, v in dim_entry["outcome_counts"].items()
            }

        relevant_n = total - dim_entry["outcome_counts"].get(NOT_RELEVANT, 0)
        if relevant_n > 0:
            dim_entry["outcome_pct_of_relevant"] = {
                k: round(v / relevant_n, 6)
                for k, v in dim_entry["outcome_counts"].items()
                if k != NOT_RELEVANT
            }

        for outcome in FAILURE_OUTCOMES:
            denom_outcome = int((df[dim] == outcome).sum())
            buckets: List[Dict[str, Any]] = []
            keys = [(d, o, q) for (d, o, q) in q_counts if d == dim and o == outcome]
            keys.sort(key=lambda k: q_counts[k], reverse=True)
            for _d, _o, qid in keys[:top_questions_per_band]:
                cnt = q_counts[(dim, outcome, qid)]
                rub = rubric_map.get(qid, {})
                pct_all = cnt / total if total else 0.0
                pct_within = cnt / denom_outcome if denom_outcome else 0.0
                item: Dict[str, Any] = {
                    "question_id": qid,
                    "count": cnt,
                    "pct_of_all_conversations": round(pct_all, 6),
                    "pct_within_outcome": round(pct_within, 6),
                    "rubric_severity": rub.get("severity", ""),
                    "rubric_dimension": rub.get("dimension", ""),
                    "question_text": rub.get("question", ""),
                }
                ex = ex_store.get((dim, outcome, qid), [])
                if ex:
                    item["exemplars"] = [
                        {"filename": fn, "judge_model": jm, "reasoning": r}
                        for fn, jm, r in ex
                    ]
                buckets.append(item)

            dim_entry["by_outcome"][outcome] = {
                "conversation_count": denom_outcome,
                "yes_questions": buckets,
            }

        out["dimensions"][dim] = dim_entry

    ordered_dims = _ordered_dimensions(dimensions, dims_present)
    overall_by_dim: List[Dict[str, Any]] = []
    for dim in ordered_dims:
        dinfo = out["dimensions"][dim]
        bands_out: List[Dict[str, Any]] = []
        for outcome in (DAMAGING, NEUTRAL):
            bo = dinfo["by_outcome"][outcome]
            yq = bo["yes_questions"]
            if not yq:
                continue
            title, sev = OUTCOME_BAND_HEADLINE[outcome]
            n_band = int(bo["conversation_count"])
            pct_band = round(n_band / total, 6) if total else 0.0
            bands_out.append(
                {
                    "outcome": outcome,
                    "band_label": title,
                    "rubric_severity": sev,
                    "conversation_count": n_band,
                    "pct_of_all_conversations": pct_band,
                    "questions": yq,
                }
            )
        if bands_out:
            overall_by_dim.append({"dimension": dim, "bands": bands_out})
    out["overall_opportunities_by_dimension"] = overall_by_dim

    global_modes: List[Dict[str, Any]] = []
    sort_key = partial(_global_failure_mode_sort_key, rubric_map=rubric_map)
    for (dim, outcome, qid), cnt in sorted(q_counts.items(), key=sort_key):
        pct_all = cnt / total if total else 0.0
        rub = rubric_map.get(qid, {})
        global_modes.append(
            {
                "dimension": dim,
                "outcome": outcome,
                "question_id": qid,
                "count": cnt,
                "pct_of_all_conversations": round(pct_all, 6),
                "question_text": rub.get("question", "") or "",
                "severity": rub.get("severity", ""),
            }
        )
    out["global_failure_modes"] = global_modes[:25]

    return out


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _render_overall_opportunities_md(data: Dict[str, Any]) -> List[str]:
    """TL;DR: one block per dimension; red (HPH) before yellow (Suboptimal)."""
    lines: List[str] = []
    overall = data.get("overall_opportunities_by_dimension") or []
    lines.append("## Overall opportunities (TL;DR)")
    lines.append("")
    if not overall:
        lines.append(
            "- No failure-mode question IDs were recorded "
            "(all Best Practice / Not Relevant, or missing `yes_question_id`)."
        )
        return lines

    lines.append(
        "Grouped by **dimension**; within each, **High Potential for Harm** "
        "(red severity) before **Suboptimal but Low Potential for Harm** "
        "(yellow severity), matching the judge rubric."
    )
    lines.append("")
    for block in overall:
        dim = block["dimension"]
        lines.append(f"**{dim}**")
        lines.append("")
        for band in block["bands"]:
            title = band["band_label"]
            sev = band["rubric_severity"]
            bpct = _fmt_pct(float(band.get("pct_of_all_conversations", 0)))
            bn = int(band.get("conversation_count", 0))
            lines.append(
                f"- **{title}** ({sev} severity) ({bpct} of all conversations; n={bn})"
            )
            for q in band["questions"]:
                qid = q["question_id"]
                pct = _fmt_pct(q["pct_of_all_conversations"])
                n = q["count"]
                lines.append(
                    f"  - **Question {qid}** ({pct} of all conversations; n={n})"
                )
                qt = (q.get("question_text") or "").strip()
                if qt:
                    for ql in (ln.strip() for ln in qt.splitlines() if ln.strip()):
                        lines.append(f"    - {ql}")
                else:
                    lines.append("    - _(No rubric question text matched this ID.)_")
            lines.append("")
        lines.append("")
    lines.append(
        "_Per-dimension sections below add `pct_within_outcome`, counts by band, "
        "and judge exemplars._"
    )
    return lines


def render_markdown(data: Dict[str, Any], dimensions_order: List[str]) -> str:
    lines: List[str] = []
    meta = data["meta"]
    total = meta["total_conversations"]
    lines.append("# Improvement report from judge results")
    lines.append("")
    if meta.get("low_sample"):
        thr = meta.get("low_sample_threshold", LOW_SAMPLE_THRESHOLD)
        lines.append(
            f"> **Low sample:** Only **{total}** conversation(s) in this file "
            f"(threshold `< {thr}`). Percentages are **noisy**; "
            "use this as directional only."
        )
        lines.append("")
    lines.append(f"- Judged conversations: **{total}**")
    lines.append("")

    lines.extend(_render_overall_opportunities_md(data))
    lines.append("")

    lines.append("## By dimension")
    lines.append("")

    dim_scores = data.get("dimension_scores", {})
    dims_in = meta.get("dimensions_in_results", dimensions_order)
    ordered = _ordered_dimensions(dimensions_order, list(dims_in))

    for dim in ordered:
        scores = dim_scores.get(dim, {})
        lines.append(f"### {dim}")
        lines.append("")
        if scores:
            tc = scores.get("total_count", 0)
            lines.append(
                f"- **Relevant conversations** (excluding Not Relevant): **{tc}**"
            )
            bp = _fmt_pct(scores.get("best_practice_pct", 0) / 100.0)
            neu = _fmt_pct(scores.get("neutral_pct", 0) / 100.0)
            dmg = _fmt_pct(scores.get("damaging_pct", 0) / 100.0)
            lines.append(
                f"- **Best Practice** {bp} · **Suboptimal** {neu} · "
                f"**High potential for harm** {dmg}"
            )
            lines.append("")

        dinfo = data["dimensions"].get(dim, {})

        for band, label in (
            (DAMAGING, "High Potential for Harm"),
            (NEUTRAL, "Suboptimal (low harm potential)"),
        ):
            bo = dinfo.get("by_outcome", {}).get(band, {})
            nband = bo.get("conversation_count", 0)
            lines.append(f"#### {label}")
            lines.append("")
            if nband == 0:
                lines.append("_No conversations in this band._")
                lines.append("")
                continue
            lines.append(
                f"_{nband} conversation(s) in this band "
                f"({_fmt_pct(nband / total) if total else '0%'} of all rows)._"
            )
            lines.append("")
            y_questions = bo.get("yes_questions", [])
            if not y_questions:
                lines.append(
                    f"_No cited rubric question IDs in this band (missing or invalid "
                    f"`{dim}_yes_question_id`)._"
                )
                lines.append("")
                continue
            for item in y_questions:
                qid = item["question_id"]
                p_all = _fmt_pct(item["pct_of_all_conversations"])
                p_band = _fmt_pct(item["pct_within_outcome"])
                lines.append(
                    f"- **Question {qid}** — {p_all} of **all** conversations; "
                    f"{p_band} of failures in this band"
                )
                lines.append(f"  - Count: **{item['count']}**")
                if item.get("rubric_severity"):
                    lines.append(f"  - Severity: **{item['rubric_severity']}**")
                qt = item.get("question_text") or ""
                if qt:
                    short = qt[:900] + ("…" if len(qt) > 900 else "")
                    lines.append(f"  - **Rubric:** {short}")
                for ex in item.get("exemplars", [])[:2]:
                    fn = ex.get("filename") or "(unknown file)"
                    jm = (ex.get("judge_model") or "").strip()
                    r = ex.get("reasoning", "")
                    rshort = (r[:500] + "…") if len(r) > 500 else r
                    who = f"`{jm}` — `{fn}`" if jm else f"`{fn}`"
                    lines.append(f"  - **Judge** ({who}): _{rshort}_")
                lines.append("")
            lines.append("")

        lines.append(f"#### Overall notes for {dim}")
        lines.append("")
        lines.append(
            "- Prioritize **High Potential for Harm** items first, then frequent "
            "**Suboptimal** rubric branches."
        )
        lines.append(
            "- Each line lists **`judge_model`** then **`filename`** so pooled or "
            "multi-judge rows for the same conversation are not confused."
        )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "**How to read percentages:** values labeled “of all conversations” use "
        "every CSV row as the denominator; “of failures in this band” is "
        "`count / conversations in that band` for that dimension."
    )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Summarize results.csv into stats JSON + Markdown report."
    )
    p.add_argument("--results", type=Path, required=True, help="Path to results.csv")
    p.add_argument("--rubric", type=Path, default=Path("data/rubric.tsv"))
    p.add_argument("--out-stats", type=Path, help="Write structured JSON here")
    p.add_argument("--out-md", type=Path, help="Write Markdown report here")
    p.add_argument(
        "--top-questions",
        type=int,
        default=12,
        help="Max questions per outcome band per dimension",
    )
    p.add_argument(
        "--exemplars",
        type=int,
        default=3,
        help="Max distinct exemplar reasonings per bucket",
    )
    p.add_argument(
        "--low-sample-threshold",
        type=int,
        default=LOW_SAMPLE_THRESHOLD,
        help="If total rows < this, set meta.low_sample and banner the Markdown report",
    )
    args = p.parse_args()

    results_path = args.results.expanduser().resolve()
    if not results_path.is_file():
        raise SystemExit(f"Results file not found: {results_path}")

    rubric_path = args.rubric.expanduser().resolve()
    if not rubric_path.is_file():
        raise SystemExit(f"Rubric file not found: {rubric_path}")

    usecols = _collect_usecols(results_path, DIMENSIONS)
    df = pd.read_csv(results_path, usecols=lambda c: c in usecols)

    rubric_map = load_rubric_question_map(rubric_path)
    data = aggregate_improvements(
        df,
        rubric_map,
        DIMENSIONS,
        exemplars_per_bucket=args.exemplars,
        top_questions_per_band=args.top_questions,
        low_sample_threshold=args.low_sample_threshold,
    )
    data["meta"]["results_path"] = str(results_path)
    data["meta"]["rubric_path"] = str(rubric_path)

    if args.out_stats:
        args.out_stats.parent.mkdir(parents=True, exist_ok=True)
        args.out_stats.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(data, DIMENSIONS), encoding="utf-8")

    if not args.out_stats and not args.out_md:
        print(json.dumps(data["meta"], indent=2))
        print(render_markdown(data, DIMENSIONS)[:4000])


if __name__ == "__main__":
    main()
