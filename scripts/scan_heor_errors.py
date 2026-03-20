#!/usr/bin/env python3
"""Scan HEOR conversations for LLM errors and group by set key.

Set keys are derived from score_combo_sets.txt:
- provider_user_turns_runs (e.g. grok4_opus45_t100_r1)
- Maps folder pattern p_{user}__a_{provider}__t{turns}__r{runs} to set keys.
"""

import hashlib
import json
import re
from pathlib import Path

CONV_BASE = Path("/Users/josh.gieringer/Projects/VERA-MH/conversations/HEOR_OLD_MODELS")
out_path = Path(__file__).resolve().parent / "heor_old_models_error_files_by_set.json"

# Error patterns to detect
ERROR_PATTERNS = [r": Error generating response"]


# Raw folder model name -> display name for role labels
MODEL_DISPLAY = {
    "gpt_5_2": "gpt5.2",
    "claude_opus_4_5_20251101": "opus45",
    "gemini_3_pro_preview": "gemini3",
    "azure_grok_4": "grok4",
}


def parse_run_folder(run_dir: Path) -> tuple[str, str, str]:
    """
    Parse run folder name. Returns (set_key, provider_llm, user_llm).
    Subfolder pattern: p_{user}__a_{provider}__t{turns}__r{runs}__{timestamp}
    """
    name = run_dir.name
    m = re.match(r"^p_(.+)__a_(.+)__t(\d+)__r(\d+)__", name)
    if not m:
        return ("unknown", "provider", "user")
    user_raw, provider_raw, turns, runs = (
        m.group(1),
        m.group(2),
        int(m.group(3)),
        int(m.group(4)),
    )
    set_key = f"provider-{provider_raw}_user-{user_raw}_t{turns}_r{runs}"
    provider_llm = MODEL_DISPLAY.get(provider_raw, provider_raw.replace("_", "-"))
    user_llm = MODEL_DISPLAY.get(user_raw, user_raw.replace("_", "-"))
    return set_key, provider_llm, user_llm


def file_has_error(content: str) -> bool:
    """Return True if content contains any of the target error patterns."""
    for pat in ERROR_PATTERNS:
        if re.search(pat, content):
            return True
    return False


def normalize_error(raw: str) -> str:
    """Normalize error string: drop request_id, replace dates with placeholder."""
    # Remove request_id (and optional preceding comma)
    out = re.sub(r",?\s*'request_id':\s*'[^']*'", "", raw)
    out = re.sub(r",?\s*\"request_id\":\s*\"[^\"]*\"", "", out)
    out = re.sub(r",\s*}", "}", out).strip()
    # Replace YYYY-MM-DD with placeholder so usage-limit errors dedupe
    # out = re.sub(r"\d{4}-\d{2}-\d{2}", "<date>", out)
    return out


# Role prefix in conversation: chatbot -> provider, user -> persona
ROLE_MAP = {"chatbot": "provider", "user": "persona"}


def extract_error_counts(content: str) -> dict[str, dict[str, int]]:
    """Extract normalized error strings and counts by LLM role (provider/persona).

    user: lines -> persona (user-agent LLM); chatbot: lines -> provider (provider LLM).
    Returns { "provider": { error: count }, "persona": { error: count } }.
    """
    counts: dict[str, dict[str, int]] = {"provider": {}, "persona": {}}
    for m in re.finditer(
        r"^(user|chatbot): Error generating response: (.+?)(?:\n|$)",
        content,
        re.MULTILINE,
    ):
        role = ROLE_MAP.get(m.group(1), "persona")
        raw = m.group(2).strip()
        norm = normalize_error(raw)
        if norm:
            counts[role][norm] = counts[role].get(norm, 0) + 1
    return counts


def merge_role_counts(
    target: dict[str, dict[str, int]],
    source: dict[str, dict[str, int]],
    provider_llm: str,
    user_llm: str,
) -> None:
    """Merge source role counts into target in-place. Uses provider-{llm} and user-{llm} keys."""
    role_keys = {"provider": f"provider-{provider_llm}", "persona": f"user-{user_llm}"}
    for role, key in role_keys.items():
        for err, cnt in source.get(role, {}).items():
            target.setdefault(key, {})
            target[key][err] = target[key].get(err, 0) + cnt


def scan(
    base: Path,
) -> tuple[
    dict[str, list[str]],
    dict[str, int],
    dict[str, dict[str, dict[str, int]]],
    dict[str, dict[str, int]],
]:
    """Scan base for .txt files. Returns (files_by_set, total_files_by_set, unique_errors_by_set, unique_errors_global).

    total_files_by_set: total .txt files per set (with or without errors)
    unique_errors_by_set: { set_key: { "provider-grok4": {error: count}, "user-opus45": {...} } }
    unique_errors_global: { "provider-grok4": {...}, "user-opus45": {...}, ... }
    """
    errors_by_set: dict[str, list[str]] = {}
    total_files_by_set: dict[str, int] = {}
    unique_errors_by_set: dict[str, dict[str, dict[str, int]]] = {}
    unique_errors_global: dict[str, dict[str, int]] = {}
    for txt_path in base.rglob("*.txt"):
        rel = str(txt_path.relative_to(base))
        run_dir = txt_path.parent
        set_key, provider_llm, user_llm = parse_run_folder(run_dir)
        total_files_by_set[set_key] = total_files_by_set.get(set_key, 0) + 1
        try:
            content = txt_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if file_has_error(content):
            errors_by_set.setdefault(set_key, []).append(rel)
            file_counts = extract_error_counts(content)
            if set_key not in unique_errors_by_set:
                unique_errors_by_set[set_key] = {}
            merge_role_counts(
                unique_errors_by_set[set_key], file_counts, provider_llm, user_llm
            )
            merge_role_counts(unique_errors_global, file_counts, provider_llm, user_llm)
    return errors_by_set, total_files_by_set, unique_errors_by_set, unique_errors_global


def find_duplicate_txt_files(base: Path) -> dict[str, list[Path]]:
    """Group all .txt files under base by content hash. Returns only groups with 2+ files."""
    content_to_paths: dict[str, list[Path]] = {}
    for txt_path in base.rglob("*.txt"):
        try:
            content = txt_path.read_bytes()
        except OSError:
            continue
        h = hashlib.sha256(content).hexdigest()
        content_to_paths.setdefault(h, []).append(txt_path)
    return {h: paths for h, paths in content_to_paths.items() if len(paths) > 1}


def main() -> None:
    if not CONV_BASE.exists():
        print(f"Base path does not exist: {CONV_BASE}")
        return

    errors_by_set, total_files_by_set, unique_errors_by_set, unique_errors_global = (
        scan(CONV_BASE)
    )

    # Counts per set key
    total = sum(len(v) for v in errors_by_set.values())
    all_global_errors = set()
    for role_errors in unique_errors_global.values():
        all_global_errors |= set(role_errors.keys())
    unique_error_count = len(all_global_errors)
    print("Counts per set key (conversations with errors):")
    print("-" * 50)
    for key in sorted(errors_by_set.keys()):
        count = len(errors_by_set[key])
        total_files = total_files_by_set.get(key, 0)
        pct = (100.0 * count / total_files) if total_files else 0.0
        by_role = unique_errors_by_set.get(key, {})
        role_summary = " / ".join(
            f"{rk}={len(errs)}" for rk, errs in sorted(by_role.items()) if errs
        )
        print(
            f"  {key}: {count}/{total_files} files ({pct:.1f}%), {role_summary} unique error type(s)"
        )
    print("-" * 50)
    print(f"  TOTAL: {total}")
    print(f"  UNIQUE ERROR TYPES (global): {unique_error_count}")

    # Unique errors per set by role (provider-X / user-X)
    print("\nUnique errors per set key (provider-LLM / user-LLM):")
    print("-" * 50)
    for key in sorted(unique_errors_by_set.keys()):
        by_role = unique_errors_by_set[key]
        print(f"\n  {key}:")
        for role_key in sorted(by_role.keys()):
            err_counts = by_role[role_key]
            if not err_counts:
                continue
            print(f"    [{role_key}]")
            for e, cnt in sorted(err_counts.items(), key=lambda x: -x[1]):
                display = e if len(e) <= 110 else e[:107] + "..."
                print(f"      - [{cnt}] {display}")

    # Duplicate .txt files (same content, different paths)
    duplicates = find_duplicate_txt_files(CONV_BASE)
    if duplicates:
        print("\nDuplicate .txt files (same content):")
        print("-" * 50)
        for content_hash, paths in sorted(duplicates.items(), key=lambda x: -len(x[1])):
            rel_paths = [str(p.relative_to(CONV_BASE)) for p in paths]
            print(f"  [{len(paths)} copies]")
            for loc in sorted(rel_paths):
                print(f"    {loc}")
        print("-" * 50)
        print(f"  {len(duplicates)} duplicate group(s)")
    else:
        print("\nNo duplicate .txt files found.")

    # Write full results to JSON
    files_with_errors_count_by_set: dict[str, dict[str, float | int]] = {}
    for set_key in total_files_by_set:
        total_files = total_files_by_set[set_key]
        error_file_count = len(errors_by_set.get(set_key, []))
        pct = (100.0 * error_file_count / total_files) if total_files else 0.0
        files_with_errors_count_by_set[set_key] = {
            "error_file_count": error_file_count,
            "total_files": total_files,
            "pct_files_w_error": round(pct, 1),
        }
    payload = {
        "total_files_with_errors": total,
        "files_with_errors_count_by_set": files_with_errors_count_by_set,
        "files_with_errors_by_set": errors_by_set,
        "total_errors_per_llm_by_set": unique_errors_by_set,
        "total_errors_per_llm_global": {
            rk: dict(sorted(errs.items(), key=lambda x: -x[1]))
            for rk, errs in sorted(unique_errors_global.items())
        },
        "unique_error_count": unique_error_count,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nFull results (files + unique errors) written to: {out_path}")

    total_txt = sum(total_files_by_set.values())
    print(f"Total .txt files examined: {total_txt}")


if __name__ == "__main__":
    main()
