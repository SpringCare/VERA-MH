#!/usr/bin/env python3
"""
Run detect_persona_role_reversal.py on two conversation directories and print
a side-by-side stats report to compare role-reversal rates.

Usage:
    uv run python3 scripts/compare_role_reversal.py [options]

The script saves intermediate CSVs so you can re-run the report without
re-classifying (use --skip-detection if both CSVs already exist).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

_REPO_ROOT = Path(__file__).resolve().parent.parent

DIRS = {
    "freaky-friday": _REPO_ROOT / "conversations" / "001-freaky-friday-role-reversal",
    "v1.1": _REPO_ROOT / "conversations" / "001-v1.1-role-reversal",
}

DEFAULT_OUT_DIR = _REPO_ROOT / "scripts"


# ---------------------------------------------------------------------------
# Detection runner
# ---------------------------------------------------------------------------


def run_detection(
    label: str,
    root: Path,
    output_csv: Path,
    model: str,
    max_concurrent: int,
    extra_args: list[str],
) -> None:
    script = _REPO_ROOT / "scripts" / "detect_persona_role_reversal.py"
    cmd = [
        "uv",
        "run",
        "python3",
        str(script),
        "--root",
        str(root),
        "--output",
        str(output_csv),
        "--model",
        model,
        "--max-concurrent",
        str(max_concurrent),
        *extra_args,
    ]
    print(f"\n{'=' * 70}", flush=True)
    print(f"[{label}] Running detection on: {root}", flush=True)
    print(f"  Output: {output_csv}", flush=True)
    print(f"  Command: {' '.join(cmd)}", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    if result.returncode != 0:
        print(
            f"[{label}] Detection script exited with code {result.returncode}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def pct(n: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{n / total * 100:.1f}%"


def fmt_p(p: float) -> str:
    """Format a p-value for display."""
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return f"p={p:.3f}"
    return f"p={p:.2f}"


def fisher_p(rev_a: int, total_a: int, rev_b: int, total_b: int) -> str:
    """Two-sided Fisher's exact test on a 2x2 contingency table.
    Returns formatted p-value string, or '' if totals are zero."""
    if total_a == 0 or total_b == 0:
        return ""
    table = [
        [rev_a, total_a - rev_a],
        [rev_b, total_b - rev_b],
    ]
    _, p_val = fisher_exact(table, alternative="two-sided")
    return fmt_p(float(p_val))


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    df["role_reversal_bool"] = df["role_reversal"].str.strip().str.lower() == "true"
    return df


def conversation_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per conversation file; True if any turn is role_reversal."""
    return (
        df.groupby("conversation_filename")["role_reversal_bool"]
        .any()
        .reset_index()
        .rename(columns={"role_reversal_bool": "has_reversal"})
    )


def breakdown_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-conversation reversal rate grouped by a metadata column."""
    conv = df.groupby(["conversation_filename", col])["role_reversal_bool"].any()
    conv = conv.reset_index().rename(columns={"role_reversal_bool": "has_reversal"})
    grp = conv.groupby(col)["has_reversal"].agg(["sum", "count"])
    grp.columns = pd.Index(["reversal_convos", "total_convos"])
    grp = grp.copy()
    grp["rate"] = grp["reversal_convos"] / grp["total_convos"]
    grp = grp.sort_values("rate", ascending=False)
    return grp


def md_table(df: pd.DataFrame, label_a: str, label_b: str) -> list[str]:
    """Return Markdown table lines for a merged breakdown DataFrame."""
    headers = [
        "Label",
        f"{label_a} — Reversals",
        f"{label_a} — Total",
        f"{label_a} — Rate",
        f"{label_b} — Reversals",
        f"{label_b} — Total",
        f"{label_b} — Rate",
        "Δ (pp)",
        "p-value",
    ]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for idx, row in df.iterrows():
        ra = int(row["reversal_convos_a"])
        ta = int(row["total_convos_a"])
        rb = int(row["reversal_convos_b"])
        tb = int(row["total_convos_b"])
        rate_a = f"{ra / ta * 100:.1f}%" if ta else "N/A"
        rate_b = f"{rb / tb * 100:.1f}%" if tb else "N/A"
        delta_str = f"{(rb / tb - ra / ta) * 100:+.1f}pp" if ta and tb else ""
        p_str = fisher_p(ra, ta, rb, tb)
        cells = [
            str(idx),
            str(ra),
            str(ta),
            rate_a,
            str(rb),
            str(tb),
            rate_b,
            delta_str,
            p_str,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def merge_breakdowns(bd_a: pd.DataFrame, bd_b: pd.DataFrame) -> pd.DataFrame:
    merged = bd_a.add_suffix("_a").join(bd_b.add_suffix("_b"), how="outer").fillna(0)
    for col in [
        "reversal_convos_a",
        "total_convos_a",
        "reversal_convos_b",
        "total_convos_b",
    ]:
        merged[col] = merged[col].astype(int)
    return merged


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(
    label_a: str, df_a: pd.DataFrame, label_b: str, df_b: pd.DataFrame
) -> str:
    """Build the full Markdown report and return it as a string."""
    lines: list[str] = []

    lines += [
        "# Role Reversal Comparison Report",
        "",
        "| | |",
        "|---|---|",
        f"| **A** | `{label_a}` |",
        f"| **B** | `{label_b}` |",
        "",
    ]

    # --- Top-level message stats ---
    ma = len(df_a)
    mb = len(df_b)
    ra_msg = int(df_a["role_reversal_bool"].sum())
    rb_msg = int(df_b["role_reversal_bool"].sum())
    pct_a = ra_msg / ma * 100 if ma else 0
    pct_b = rb_msg / mb * 100 if mb else 0

    msg_p = fisher_p(ra_msg, ma, rb_msg, mb)
    msg_rate_row = (
        f"| Rate | {pct_a:.1f}% | {pct_b:.1f}% | {pct_b - pct_a:+.1f}pp | {msg_p} |"
    )
    rr_row = (
        f"| `role_reversal=true` messages "
        f"| {ra_msg:,} | {rb_msg:,} | {rb_msg - ra_msg:+,} | |"
    )
    lines += [
        "## Message-Level",
        "",
        f"| Metric | {label_a} | {label_b} | Δ (B−A) | p-value |",
        "|---|---|---|---|---|",
        f"| Total user messages classified | {ma:,} | {mb:,} | {mb - ma:+,} | |",
        rr_row,
        msg_rate_row,
        "",
    ]

    # --- Conversation-level ---
    conv_a = conversation_level(df_a)
    conv_b = conversation_level(df_b)
    ca_tot = len(conv_a)
    cb_tot = len(conv_b)
    ca_rev = int(conv_a["has_reversal"].sum())
    cb_rev = int(conv_b["has_reversal"].sum())
    cpct_a = ca_rev / ca_tot * 100 if ca_tot else 0
    cpct_b = cb_rev / cb_tot * 100 if cb_tot else 0
    conv_p = fisher_p(ca_rev, ca_tot, cb_rev, cb_tot)

    if cpct_b < cpct_a:
        verdict = (
            f"✅ **{label_b}** has **fewer** reversals ({cpct_b:.1f}% vs {cpct_a:.1f}%)"
        )
    elif cpct_b > cpct_a:
        verdict = (
            f"❌ **{label_b}** has **more** reversals ({cpct_b:.1f}% vs {cpct_a:.1f}%)"
        )
    else:
        verdict = "➡️ No difference in conversation-level reversal rate"

    conv_rev_row = (
        f"| Conversations with ≥1 reversal "
        f"| {ca_rev:,} | {cb_rev:,} | {cb_rev - ca_rev:+,} | |"
    )
    conv_rate_row = (
        f"| Rate | {cpct_a:.1f}% | {cpct_b:.1f}% "
        f"| {cpct_b - cpct_a:+.1f}pp | {conv_p} |"
    )
    lines += [
        "## Conversation-Level",
        "",
        f"| Metric | {label_a} | {label_b} | Δ (B−A) | p-value |",
        "|---|---|---|---|---|",
        f"| Total conversations | {ca_tot:,} | {cb_tot:,} | {cb_tot - ca_tot:+,} | |",
        conv_rev_row,
        conv_rate_row,
        "",
        f"> **Verdict:** {verdict}",
        "",
    ]

    # --- Breakdown by user model ---
    bd_a_model = breakdown_by(df_a, "user_model_slug")
    bd_b_model = breakdown_by(df_b, "user_model_slug")
    merged_model = merge_breakdowns(bd_a_model, bd_b_model)
    lines += ["## By User Model", ""]
    lines += md_table(merged_model, label_a, label_b)
    lines.append("")

    # --- Breakdown by risk level ---
    bd_a_risk = breakdown_by(df_a, "short_suicide_risk_level")
    bd_b_risk = breakdown_by(df_b, "short_suicide_risk_level")
    merged_risk = merge_breakdowns(bd_a_risk, bd_b_risk)
    lines += ["## By Suicide Risk Level", ""]
    lines += md_table(merged_risk, label_a, label_b)
    lines.append("")

    # --- Breakdown by persona ---
    bd_a_persona = breakdown_by(df_a, "persona_name")
    bd_b_persona = breakdown_by(df_b, "persona_name")
    merged_persona = merge_breakdowns(bd_a_persona, bd_b_persona)
    lines += ["## By Persona", ""]
    lines += md_table(merged_persona, label_a, label_b)
    lines.append("")

    # --- Errors ---
    err_a = df_a["error"].notna() & (df_a["error"].str.strip() != "")
    err_b = df_b["error"].notna() & (df_b["error"].str.strip() != "")
    if err_a.any() or err_b.any():
        lines += [
            "## Errors",
            "",
            "| | Count |",
            "|---|---|",
            f"| {label_a} errors | {int(err_a.sum())} |",
            f"| {label_b} errors | {int(err_b.sum())} |",
            "",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare role-reversal rates across two conversation directories"
    )
    p.add_argument(
        "--dir-a",
        type=Path,
        default=DIRS["v1.1"],
        help="First conversation directory (default: 001-v1.1-role-reversal)",
    )
    p.add_argument(
        "--label-a",
        default="v1.1",
        help="Label for directory A in the report",
    )
    p.add_argument(
        "--dir-b",
        type=Path,
        default=DIRS["freaky-friday"],
        help="Second conversation directory (default: 001-freaky-friday-role-reversal)",
    )
    p.add_argument(
        "--label-b",
        default="freaky-friday",
        help="Label for directory B in the report",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for intermediate CSVs (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--model",
        "-m",
        default="gpt-4o-mini",
        help="Classifier model (default: gpt-4o-mini)",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=12,
        help="Max concurrent API calls per run (default: 12)",
    )
    p.add_argument(
        "--skip-detection",
        action="store_true",
        help="Skip running detect_persona_role_reversal.py; load existing CSVs only",
    )
    p.add_argument(
        "--classify-all-user-messages",
        action="store_true",
        help="Pass --classify-all-user-messages to the detection script",
    )
    p.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Limit files per run (debug)",
    )
    p.add_argument(
        "--limit-messages",
        type=int,
        default=None,
        help="Limit messages per file (debug)",
    )
    p.add_argument(
        "--markdown-output",
        "-md",
        type=Path,
        default=None,
        help="Write the report as a Markdown file to this path (in addition to stdout)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    csv_a = args.out_dir / f"role_reversal_{args.label_a.replace('/', '_')}.csv"
    csv_b = args.out_dir / f"role_reversal_{args.label_b.replace('/', '_')}.csv"

    extra: list[str] = []
    if args.classify_all_user_messages:
        extra.append("--classify-all-user-messages")
    if args.limit_files is not None:
        extra += ["--limit-files", str(args.limit_files)]
    if args.limit_messages is not None:
        extra += ["--limit-messages", str(args.limit_messages)]

    if not args.skip_detection:
        run_detection(
            args.label_a, args.dir_a, csv_a, args.model, args.max_concurrent, extra
        )
        run_detection(
            args.label_b, args.dir_b, csv_b, args.model, args.max_concurrent, extra
        )
    else:
        print("\n[--skip-detection] Loading existing CSVs:")
        print(f"  A: {csv_a}")
        print(f"  B: {csv_b}")

    for path, label in [(csv_a, args.label_a), (csv_b, args.label_b)]:
        if not path.exists():
            print(
                f"ERROR: CSV not found for {label}: {path}\n"
                "Run without --skip-detection to generate it.",
                file=sys.stderr,
            )
            sys.exit(1)

    df_a = load_and_clean(csv_a)
    df_b = load_and_clean(csv_b)

    report = build_report(args.label_a, df_a, args.label_b, df_b)
    print(report)

    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(report, encoding="utf-8")
        print(f"\nMarkdown report written to: {args.markdown_output}", file=sys.stderr)

    print("\nIntermediate CSVs:", file=sys.stderr)
    print(f"  A: {csv_a}", file=sys.stderr)
    print(f"  B: {csv_b}", file=sys.stderr)


if __name__ == "__main__":
    main()
