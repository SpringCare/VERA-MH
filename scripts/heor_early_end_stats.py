#!/usr/bin/env python3
"""Scan HEOR conversations for early endings and report stats.

Usage:
    python3 scripts/heor_early_end_stats.py [--root conversations/HEOR] \\
        [--json] [--tsv] [--patterns-file path] [--plots DIR]
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False

END_MARKER = "[CONVERSATION ENDED - persona signaled termination]"
END_OF_CONV = "<END OF CONVERSATION>"
SET_DIR_PATTERN = re.compile(r"^Set_(\d+)-(.+?)-Persona_(.+?)-(\d+)x(\d+)$")
FILENAME_PATTERN = re.compile(r"^[a-f0-9]{6}_([^_]+)_.+_run\d+\.txt$", re.IGNORECASE)
TURN_PATTERN = re.compile(r"^(user|chatbot):", re.MULTILINE)
WORD_PATTERN = re.compile(r"[a-zA-Z']+")
TOP_N_WORDS = 30
TOP_N_BIGRAMS = 20
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "i",
    "im",
    "you",
    "your",
    "this",
    "but",
    "they",
    "we",
    "or",
    "if",
    "so",
}


def parse_set_dir(name: str) -> Optional[tuple[str, str, str, int]]:
    """Parse set dir name -> (set_id, provider_llm, persona_llm, max_turns)."""
    m = SET_DIR_PATTERN.match(name)
    if not m:
        return None
    set_id, provider_seg, persona_llm, _n, max_turns = m.groups()
    provider_llm = (
        provider_seg.replace("Provider_", "", 1)
        if provider_seg.startswith("Provider_")
        else provider_seg
    )
    return set_id, provider_llm, persona_llm, int(max_turns)


def parse_filename(name: str) -> Optional[str]:
    """Extract persona name from filename. Returns None if pattern doesn't match."""
    m = FILENAME_PATTERN.match(name)
    return m.group(1) if m else None


def count_turns(content: str) -> int:
    """Count message blocks (each user: or chatbot: line start = one turn)."""
    return len(TURN_PATTERN.findall(content))


def extract_last_user_message(content: str) -> Optional[str]:
    """Extract last user block with <END OF CONVERSATION>. Strip markers."""
    if END_OF_CONV not in content or END_MARKER not in content:
        return None
    blocks = re.split(r"\n(?=user:|chatbot:)", content)
    for block in reversed(blocks):
        if block.strip().startswith("user:") and END_OF_CONV in block:
            text = block.replace("user:", "", 1).strip()
            text = re.sub(
                r"\s*<END OF CONVERSATION>\s*", " ", text, flags=re.IGNORECASE
            )
            text = re.sub(r"\s*\[CONVERSATION ENDED[^\]]*\]\s*", " ", text).strip()
            return text or None
    return None


def scan_file(
    path: Path,
    set_id: str,
    provider_llm: str,
    persona_llm: str,
    max_turns: int,
    persona: str,
) -> dict[str, Any]:
    """Read one conversation file; return record with ended_early, turn_ended, etc."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return {
            "path": str(path),
            "set_id": set_id,
            "provider_llm": provider_llm,
            "persona_llm": persona_llm,
            "persona": persona,
            "max_turns": max_turns,
            "ended_early": False,
            "turn_ended": 0,
            "last_user_message": None,
            "error": True,
        }
    ended_early = END_MARKER in content
    turn_ended = count_turns(content)
    last_user_message = extract_last_user_message(content) if ended_early else None
    return {
        "path": str(path),
        "set_id": set_id,
        "provider_llm": provider_llm,
        "persona_llm": persona_llm,
        "persona": persona,
        "max_turns": max_turns,
        "ended_early": ended_early,
        "turn_ended": turn_ended,
        "last_user_message": last_user_message,
        "error": False,
    }


def scan_root(root: Path) -> list[dict[str, Any]]:
    """Scan all set dirs under root and return list of file records."""
    records: list[dict[str, Any]] = []
    if not root.is_dir():
        return records
    for set_dir in sorted(root.iterdir()):
        if not set_dir.is_dir():
            continue
        parsed = parse_set_dir(set_dir.name)
        if parsed is None:
            continue
        set_id, provider_llm, persona_llm, max_turns = parsed
        for path in set_dir.glob("*.txt"):
            persona = parse_filename(path.name) or "unknown"
            rec = scan_file(path, set_id, provider_llm, persona_llm, max_turns, persona)
            records.append(rec)
    return records


def word_freq(messages: list[str], stop: bool = True) -> Counter:
    """Count words across messages; optionally exclude stopwords."""
    c: Counter = Counter()
    for msg in messages:
        for word in WORD_PATTERN.findall(msg.lower()):
            if stop and word in STOPWORDS:
                continue
            c[word] += 1
    return c


def bigrams(messages: list[str]) -> Counter:
    """Count bigrams (two consecutive words) across messages, normalized."""
    c: Counter = Counter()
    for msg in messages:
        words = WORD_PATTERN.findall(msg.lower())
        for i in range(len(words) - 1):
            if words[i] in STOPWORDS and words[i + 1] in STOPWORDS:
                continue
            c[(words[i], words[i + 1])] += 1
    return c


def build_aggregates(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build counts and rates for all breakdowns."""
    early = [r for r in records if r.get("ended_early") and not r.get("error")]
    total_files = len([r for r in records if not r.get("error")])
    total_early = len(early)

    def denom(key_fn):
        from collections import defaultdict

        d: dict[tuple, int] = defaultdict(int)
        for r in records:
            if r.get("error"):
                continue
            k = key_fn(r)
            if isinstance(k, (list, tuple)):
                d[tuple(k)] += 1
            else:
                d[(k,)] += 1
        return dict(d)

    def early_count(key_fn):
        from collections import defaultdict

        d: dict[tuple, int] = defaultdict(int)
        for r in early:
            k = key_fn(r)
            if isinstance(k, (list, tuple)):
                d[tuple(k)] += 1
            else:
                d[(k,)] += 1
        return dict(d)

    by_persona_denom = denom(lambda r: r["persona"])
    by_persona_early = early_count(lambda r: r["persona"])
    by_user_llm_denom = denom(lambda r: r["persona_llm"])
    by_user_llm_early = early_count(lambda r: r["persona_llm"])
    by_user_persona_denom = denom(lambda r: (r["persona_llm"], r["persona"]))
    by_user_persona_early = early_count(lambda r: (r["persona_llm"], r["persona"]))
    by_provider_denom = denom(lambda r: r["provider_llm"])
    by_provider_early = early_count(lambda r: r["provider_llm"])
    by_provider_persona_denom = denom(lambda r: (r["provider_llm"], r["persona"]))
    by_provider_persona_early = early_count(lambda r: (r["provider_llm"], r["persona"]))
    by_provider_user_denom = denom(lambda r: (r["provider_llm"], r["persona_llm"]))
    by_provider_user_early = early_count(
        lambda r: (r["provider_llm"], r["persona_llm"])
    )
    by_full_denom = denom(lambda r: (r["provider_llm"], r["persona_llm"], r["persona"]))
    by_full_early = early_count(
        lambda r: (r["provider_llm"], r["persona_llm"], r["persona"])
    )
    by_set_denom = denom(lambda r: r["set_id"])
    by_set_early = early_count(lambda r: r["set_id"])
    by_max_turns_denom = denom(lambda r: r["max_turns"])
    by_max_turns_early = early_count(lambda r: r["max_turns"])

    turn_dist: Counter = Counter()
    for r in early:
        turn_dist[r["turn_ended"]] += 1

    last_messages = [
        r["last_user_message"] for r in early if r.get("last_user_message")
    ]
    wf = word_freq(last_messages)
    bg = bigrams(last_messages)

    return {
        "total_files": total_files,
        "total_early": total_early,
        "by_persona": _rates(by_persona_early, by_persona_denom),
        "by_llm_user": _rates(by_user_llm_early, by_user_llm_denom),
        "by_llm_user_persona": _rates(by_user_persona_early, by_user_persona_denom),
        "by_provider": _rates(by_provider_early, by_provider_denom),
        "by_provider_persona": _rates(
            by_provider_persona_early, by_provider_persona_denom
        ),
        "by_provider_user": _rates(by_provider_user_early, by_provider_user_denom),
        "by_provider_user_persona": _rates(by_full_early, by_full_denom),
        "by_set": _rates(by_set_early, by_set_denom),
        "by_max_turns": _rates(by_max_turns_early, by_max_turns_denom),
        "turn_distribution": dict(turn_dist),
        "top_words": wf.most_common(TOP_N_WORDS),
        "top_bigrams": [((a, b), c) for (a, b), c in bg.most_common(TOP_N_BIGRAMS)],
        "early_records": early,
    }


def _rates(early_dict: dict, denom_dict: dict) -> list[dict[str, Any]]:
    """Build list of {key, count, total, pct} for printing/JSON."""
    keys = sorted(set(early_dict.keys()) | set(denom_dict.keys()))
    out = []
    for k in keys:
        count = early_dict.get(k, 0)
        total = denom_dict.get(k, 0)
        pct = (100.0 * count / total) if total else 0.0
        key_display = k[0] if len(k) == 1 else k
        out.append(
            {
                "key": key_display,
                "early_count": count,
                "total": total,
                "pct": round(pct, 1),
            }
        )
    return out


def print_tables(agg: dict[str, Any]) -> None:
    """Print human-readable tables to stdout."""
    total_files = agg["total_files"]
    total_early = agg["total_early"]
    pct = (100.0 * total_early / total_files) if total_files else 0

    print("=== HEOR early-ending stats ===\n")
    print(f"Total conversations scanned: {total_files}")
    print(f"Ended early (persona signaled termination): {total_early} ({pct:.1f}%)\n")

    print("--- By set ---")
    for row in sorted(agg["by_set"], key=lambda x: str(x["key"])):
        print(f"  {row['key']}: {row['early_count']}/{row['total']} ({row['pct']}%)")
    print()

    print("--- By max_turns (20 vs 100) ---")
    for row in agg["by_max_turns"]:
        print(f"  {row['key']}: {row['early_count']}/{row['total']} ({row['pct']}%)")
    print()

    print("--- Turn distribution (early-ended only) ---")
    dist = agg["turn_distribution"]
    for turn in sorted(dist.keys()):
        print(f"  turn {turn}: {dist[turn]}")
    print()

    print("--- By persona ---")
    for row in sorted(agg["by_persona"], key=lambda x: -x["early_count"])[:25]:
        print(f"  {row['key']}: {row['early_count']}/{row['total']} ({row['pct']}%)")
    if len(agg["by_persona"]) > 25:
        print(f"  ... and {len(agg['by_persona']) - 25} more")
    print()

    print("--- By LLM as user (persona_llm) ---")
    for row in agg["by_llm_user"]:
        print(f"  {row['key']}: {row['early_count']}/{row['total']} ({row['pct']}%)")
    print()

    print("--- By LLM as user + persona ---")
    for row in sorted(agg["by_llm_user_persona"], key=lambda x: -x["early_count"])[:20]:
        k = row["key"]
        lbl = f"{k[0]} / {k[1]}" if isinstance(k, tuple) else k
        print(f"  {lbl}: {row['early_count']}/{row['total']} ({row['pct']}%)")
    if len(agg["by_llm_user_persona"]) > 20:
        print(f"  ... and {len(agg['by_llm_user_persona']) - 20} more")
    print()

    print("--- By LLM as provider ---")
    for row in agg["by_provider"]:
        print(f"  {row['key']}: {row['early_count']}/{row['total']} ({row['pct']}%)")
    print()

    print("--- By LLM as provider + persona ---")
    for row in sorted(agg["by_provider_persona"], key=lambda x: -x["early_count"])[:20]:
        k = row["key"]
        lbl = f"{k[0]} / {k[1]}" if isinstance(k, tuple) else k
        print(f"  {lbl}: {row['early_count']}/{row['total']} ({row['pct']}%)")
    if len(agg["by_provider_persona"]) > 20:
        print(f"  ... and {len(agg['by_provider_persona']) - 20} more")
    print()

    print("--- By LLM as provider + LLM as user ---")
    for row in agg["by_provider_user"]:
        k = row["key"]
        lbl = f"{k[0]} / {k[1]}" if isinstance(k, tuple) else k
        print(f"  {lbl}: {row['early_count']}/{row['total']} ({row['pct']}%)")
    print()

    print("--- By LLM as provider + LLM as user + persona ---")
    for row in sorted(agg["by_provider_user_persona"], key=lambda x: -x["early_count"])[
        :20
    ]:
        k = row["key"]
        lbl = f"{k[0]} / {k[1]} / {k[2]}" if isinstance(k, tuple) else k
        print(f"  {lbl}: {row['early_count']}/{row['total']} ({row['pct']}%)")
    if len(agg["by_provider_user_persona"]) > 20:
        print(f"  ... and {len(agg['by_provider_user_persona']) - 20} more")
    print()

    print("--- Patterns in last user message (word frequency) ---")
    for word, count in agg["top_words"]:
        print(f"  {word}: {count}")
    print()

    print("--- Patterns in last user message (bigrams) ---")
    for (a, b), count in agg["top_bigrams"]:
        print(f"  '{a} {b}': {count}")


def write_json(agg: dict[str, Any], path: Optional[Path] = None) -> None:
    """Write JSON summary; exclude early_records if large, or include for debugging."""
    out = {k: v for k, v in agg.items() if k != "early_records"}
    out["total_early"] = agg["total_early"]
    out["total_files"] = agg["total_files"]
    text = json.dumps(out, indent=2)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        print(text)


def write_tsv(agg: dict[str, Any], root: Path) -> None:
    """Write TSV files for key breakdowns under root."""
    root.mkdir(parents=True, exist_ok=True)
    tsvs = [
        (
            "heor_early_end_by_set.tsv",
            agg["by_set"],
            ["set_id", "early_count", "total", "pct"],
        ),
        (
            "heor_early_end_by_max_turns.tsv",
            agg["by_max_turns"],
            ["max_turns", "early_count", "total", "pct"],
        ),
        (
            "heor_early_end_by_persona.tsv",
            agg["by_persona"],
            ["persona", "early_count", "total", "pct"],
        ),
        (
            "heor_early_end_by_llm_user.tsv",
            agg["by_llm_user"],
            ["persona_llm", "early_count", "total", "pct"],
        ),
        (
            "heor_early_end_by_provider.tsv",
            agg["by_provider"],
            ["provider_llm", "early_count", "total", "pct"],
        ),
        (
            "heor_early_end_by_provider_user.tsv",
            agg["by_provider_user"],
            ["provider_llm", "persona_llm", "early_count", "total", "pct"],
        ),
    ]
    for fname, rows, headers in tsvs:
        path = root / fname
        with path.open("w", encoding="utf-8") as f:
            f.write("\t".join(headers) + "\n")
            for row in rows:
                k = row["key"]
                if isinstance(k, tuple):
                    parts = list(k) + [
                        str(row["early_count"]),
                        str(row["total"]),
                        str(row["pct"]),
                    ]
                else:
                    parts = [
                        str(k),
                        str(row["early_count"]),
                        str(row["total"]),
                        str(row["pct"]),
                    ]
                f.write("\t".join(parts) + "\n")


def write_patterns_file(agg: dict[str, Any], path: Path) -> None:
    """Write word counts and optional snippets to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Word frequency in last user message (early-ended conversations)",
        "",
    ]
    for word, count in agg["top_words"]:
        lines.append(f"{word}\t{count}")
    lines.extend(
        [
            "",
            "# Bigrams",
            "",
        ]
    )
    for (a, b), count in agg["top_bigrams"]:
        lines.append(f"{a} {b}\t{count}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(
    agg: dict[str, Any],
    out_dir: Path,
    *,
    agg_20: Optional[dict[str, Any]] = None,
    agg_100: Optional[dict[str, Any]] = None,
) -> None:
    """Write seaborn/matplotlib charts to out_dir. Split by max_turns when agg_20/agg_100 provided."""
    if not _PLOTTING_AVAILABLE:
        raise RuntimeError("Plotting requires matplotlib and seaborn; install them.")
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="husl")

    # 1. Turn distribution by max_turns (20 and 100)
    early = agg.get("early_records", [])
    for max_t in (20, 100):
        dist = Counter(r["turn_ended"] for r in early if r["max_turns"] == max_t)
        if not dist:
            continue
        turns = sorted(dist.keys())
        counts = [dist[t] for t in turns]
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(
            x=[str(t) for t in turns],
            y=counts,
            color="steelblue",
            ax=ax,
        )
        ax.set_xlabel("Turn when conversation ended")
        ax.set_ylabel("Count (early-ended only)")
        ax.set_title(f"Distribution of turn at early end (max {max_t} turns)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(
            out_dir / f"turn_distribution_max{max_t}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    # Plots split by max_turns (max20 and max100)
    for max_t, a in ((20, agg_20), (100, agg_100)):
        if not a:
            continue
        suffix = f"_max{max_t}"

        # By set
        rows = sorted(a.get("by_set", []), key=lambda r: str(r["key"]))
        if rows:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(
                x=[str(r["key"]) for r in rows],
                y=[r["pct"] for r in rows],
                color="coral",
                ax=ax,
            )
            ax.set_xlabel("Set")
            ax.set_ylabel("Early-end rate (%)")
            ax.set_title(f"Early-end rate by set (max {max_t} turns)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fig.savefig(out_dir / f"by_set{suffix}.png", dpi=150, bbox_inches="tight")
            plt.close()

        # By persona (top 20)
        rows = sorted(
            a.get("by_persona", []),
            key=lambda r: (-r["pct"], -r["early_count"]),
        )[:20]
        if rows:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(
                y=[r["key"] for r in rows],
                x=[r["pct"] for r in rows],
                color="mediumpurple",
                ax=ax,
                orient="h",
            )
            ax.set_xlabel("Early-end rate (%)")
            ax.set_ylabel("Persona")
            ax.set_title(f"Early-end rate by persona, top 20 (max {max_t} turns)")
            plt.tight_layout()
            fig.savefig(
                out_dir / f"by_persona{suffix}.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

        # By LLM as user
        rows = a.get("by_llm_user", [])
        if rows:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(
                x=[str(r["key"]) for r in rows],
                y=[r["pct"] for r in rows],
                color="teal",
                ax=ax,
            )
            ax.set_xlabel("LLM as user (persona)")
            ax.set_ylabel("Early-end rate (%)")
            ax.set_title(f"Early-end rate by LLM as user (max {max_t} turns)")
            plt.tight_layout()
            fig.savefig(
                out_dir / f"by_llm_user{suffix}.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

        # By LLM as provider
        rows = a.get("by_provider", [])
        if rows:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                x=[str(r["key"]) for r in rows],
                y=[r["pct"] for r in rows],
                color="darkorange",
                ax=ax,
            )
            ax.set_xlabel("LLM as provider")
            ax.set_ylabel("Early-end rate (%)")
            ax.set_title(f"Early-end rate by LLM as provider (max {max_t} turns)")
            plt.tight_layout()
            fig.savefig(
                out_dir / f"by_provider{suffix}.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

        # Heatmap: provider x persona_llm
        rows = a.get("by_provider_user", [])
        if rows and any(isinstance(r["key"], tuple) for r in rows):
            providers = sorted(
                {r["key"][0] for r in rows if isinstance(r["key"], tuple)}
            )
            users = sorted({r["key"][1] for r in rows if isinstance(r["key"], tuple)})
            rate = {(r["key"][0], r["key"][1]): r["pct"] for r in rows}
            matrix = [[rate.get((p, u), 0) for u in users] for p in providers]
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                matrix,
                xticklabels=users,
                yticklabels=providers,
                annot=True,
                fmt=".0f",
                cmap="YlOrRd",
                ax=ax,
                cbar_kws={"label": "Early-end rate (%)"},
            )
            ax.set_xlabel("LLM as user")
            ax.set_ylabel("LLM as provider")
            ax.set_title(f"Early-end rate: provider × user (max {max_t} turns)")
            plt.tight_layout()
            fig.savefig(
                out_dir / f"heatmap_provider_user{suffix}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

    # Overall: by_max_turns (20 vs 100 comparison, single plot)
    rows = agg.get("by_max_turns", [])
    if rows:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(
            x=[str(r["key"]) for r in rows],
            y=[r["pct"] for r in rows],
            color="seagreen",
            ax=ax,
        )
        ax.set_xlabel("Max turns")
        ax.set_ylabel("Early-end rate (%)")
        ax.set_title("Early-end rate: 20 vs 100 turns")
        plt.tight_layout()
        fig.savefig(out_dir / "by_max_turns.png", dpi=150, bbox_inches="tight")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="HEOR early-ending stats")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("conversations/HEOR"),
        help="Root HEOR directory",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write JSON summary to file",
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        default=None,
        metavar="DIR",
        help="Write TSV files to directory",
    )
    parser.add_argument(
        "--patterns-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write word/bigram counts to file",
    )
    parser.add_argument(
        "--plots",
        type=Path,
        default=None,
        metavar="DIR",
        help="Write seaborn/matplotlib charts to directory",
    )
    args = parser.parse_args()

    root = args.root
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    records = scan_root(root)
    if not records:
        print("No conversation files found under", args.root)
        return

    agg = build_aggregates(records)
    agg["total_files"] = len([r for r in records if not r.get("error")])
    agg["total_early"] = len(
        [r for r in records if r.get("ended_early") and not r.get("error")]
    )

    print_tables(agg)

    def _resolve_out(path: Path) -> Path:
        """Resolve relative paths against cwd so output goes to the intended location."""
        return (path if path.is_absolute() else Path.cwd() / path).resolve()

    if args.json is not None:
        write_json(agg, _resolve_out(args.json))
    if args.tsv is not None:
        write_tsv(agg, _resolve_out(args.tsv))
    if args.patterns_file is not None:
        write_patterns_file(agg, _resolve_out(args.patterns_file))
    if args.plots is not None:
        if not _PLOTTING_AVAILABLE:
            raise SystemExit(
                "Plotting requires matplotlib and seaborn. Install: uv add seaborn"
            )
        plot_dir = _resolve_out(args.plots)
        records_ok = [r for r in records if not r.get("error")]
        records_20 = [r for r in records_ok if r.get("max_turns") == 20]
        records_100 = [r for r in records_ok if r.get("max_turns") == 100]
        agg_20 = build_aggregates(records_20) if records_20 else {}
        agg_100 = build_aggregates(records_100) if records_100 else {}
        write_plots(agg, plot_dir, agg_20=agg_20, agg_100=agg_100)
        print("Wrote plots to", plot_dir)


if __name__ == "__main__":
    main()
