"""Pool several judge evaluations into one scored result.

Merges the ``results.csv`` of several ``j_*`` evaluation folders (any mix of user
models and judge models), writes the merged CSV plus ``pool_metadata.json`` into
a new ``j_*``-style folder, and scores it like `score.run.run_scoring`. Typical
use is combining the two user-model suites of the recommended profile.

`run_pooling` is the scoring domain's application function for ``vera pool``;
``legacy/pool_vera_scores.py`` wraps the same function for the legacy CLI.

Until the Traceability phase (docs/roadmap.md) builds the target-rooted layout,
the pooled folder is named in the legacy ``j_<judge>__p_<a>+<b>__a_<agent>__...``
style and written under the caller's output folder.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from judge.score_utils import ensure_results_csv
from score.aggregate import (
    print_scores,
    save_results_json,
    score_results,
    score_results_by_risk,
)
from score.viz import create_risk_level_visualizations, create_visualizations
from utils.naming import parse_generation_run_folder_name

# ``j_{judge_info}_{timestamp}__p_...`` or ``...__conversations`` (legacy flat layout)
_JUDGE_EVAL_DIR_HEAD = re.compile(
    r"^j_(?P<judge>.+?)_\d{8}_\d{6}_\d+__(?:p_|conversations)",
    re.IGNORECASE,
)
_JUDGE_SPEC_TOKEN = re.compile(r"([^_]+x\d+)")


def _resolve_eval_input(path: Path) -> Path:
    """
    Normalize a CLI path to the judge evaluation directory (``j_*`` folder).

    Accepts either the evaluation directory itself or a ``results.csv`` file inside
    it, so users can pass ``.../j_*`` or ``.../j_*/results.csv`` interchangeably.

    Args:
        path: User-supplied filesystem path (file or directory).

    Returns:
        Resolved path to the parent ``j_*`` evaluation folder.

    Raises:
        ValueError: If *path* is a file that is not named ``results.csv``.
        FileNotFoundError: If *path* does not exist.
    """
    path = path.resolve()
    if path.is_file():
        if path.name != "results.csv":
            raise ValueError(f"Expected results.csv or a directory, got: {path}")
        return path.parent
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _generation_folder_for_eval(eval_dir: Path) -> Path | None:
    """
    Note: Only works if the evaluation folder is nested under the generation run folder.
    Walk from an evaluation folder up to the corresponding generation run folder.

    Expected layout: ``.../p_*__/evaluations/j_*`` (nested generation run). If
    *eval_dir* is not directly under an ``evaluations`` folder whose parent is a
    ``p_*`` run (e.g. legacy ``repo/evaluations/j_*``), returns None. Callers still
    pool successfully; they only lose path-derived metadata for the synthetic output
    folder name.

    Args:
        eval_dir: Resolved path to a ``j_*`` evaluation directory.

    Returns:
        Path to the ``p_*`` generation run folder, or None if the layout does not match.
    """
    eval_dir = eval_dir.resolve()
    parent = eval_dir.parent
    if parent.name != "evaluations":
        return None
    gen = parent.parent
    if gen.name.startswith("p_"):
        return gen
    return None


def _persona_slug_for_pool(source_eval_dirs: list[Path]) -> str:
    """
    Build a short ``p_*`` path segment from each source generation folder basename.

    Returns one persona model slug, or sorted unique names joined by ``+`` when
    multiple user models are merged (order-independent, stable across runs).
    """
    persona_names: list[str] = []
    for d in source_eval_dirs:
        gen = _generation_folder_for_eval(d)
        if gen is None:
            persona_names.append("unknown")
            continue
        try:
            meta = parse_generation_run_folder_name(gen.name)
            persona_names.append(str(meta["persona"]))
        except ValueError:
            persona_names.append("unknown")
    return "+".join(sorted(set(persona_names)))


def _judge_slug_from_dataframe(df: Any) -> str:
    """
    Build a judge segment like ``gpt-5.4x1`` or ``gpt-4ox2+claude-sonnet-4-5x1``.

    Matches ``judge.py`` / ``judge.runner`` folder naming (``modelx<count>`` per
    judge). Multiple judges are joined with ``+`` (stable sorted order).
    """

    if "judge_model" not in df.columns or df["judge_model"].dropna().empty:
        return "unknown"

    models = sorted(df["judge_model"].dropna().astype(str).unique())
    parts: list[str] = []
    for model in models:
        sub = df[df["judge_model"].astype(str) == model]
        if "judge_instance" in sub.columns and sub["judge_instance"].notna().any():
            count = int(sub["judge_instance"].max())
        else:
            count = 1
        parts.append(f"{model}x{count}")
    return "+".join(parts)


def _merge_judge_slug_tokens(*slugs: str) -> str:
    """Merge ``modelxN`` tokens from one or more slug strings (max count per model)."""
    counts: dict[str, int] = {}
    for slug in slugs:
        for part in slug.split("+"):
            m = re.fullmatch(r"(.+)x(\d+)", part)
            if not m:
                continue
            model, count = m.group(1), int(m.group(2))
            counts[model] = max(counts.get(model, 0), count)
    if not counts:
        return "unknown"
    return "+".join(f"{model}x{count}" for model, count in sorted(counts.items()))


def _judge_slug_from_eval_dir_name(dir_name: str) -> str | None:
    """
    Parse ``modelx<count>`` judge spec(s) from a ``j_*`` evaluation folder basename.

    Returns None for legacy ``j_pooled__*`` names or unparseable basenames.
    Multi-judge batch folders use ``_`` between specs in ``judge.py``; this returns
    ``+``-joined specs for pooled output naming.
    """
    if dir_name.startswith("j_pooled__"):
        return None
    m = _JUDGE_EVAL_DIR_HEAD.match(dir_name)
    if not m:
        return None
    tokens = _JUDGE_SPEC_TOKEN.findall(m.group("judge"))
    if not tokens:
        return None
    return "+".join(sorted(tokens))


def _judge_slug_for_pool(
    combined: Any,
    source_eval_dirs: list[Path],
    *,
    judge_slug: str | None = None,
) -> str:
    """Resolve the judge segment for a merged evaluation folder name."""
    if judge_slug:
        return judge_slug
    if "judge_model" in combined.columns and combined["judge_model"].notna().any():
        return _judge_slug_from_dataframe(combined)
    dir_slugs = [
        s for d in source_eval_dirs if (s := _judge_slug_from_eval_dir_name(d.name))
    ]
    if dir_slugs:
        return _merge_judge_slug_tokens(*dir_slugs)
    return "unknown"


def _synthetic_pooled_folder_basename(
    source_eval_dirs: list[Path],
    combined: Any,
    *,
    judge_slug: str | None = None,
) -> str:
    """
    Build a ``j_*``-style basename for the pooled output directory.

    The judge segment uses ``modelx<count>`` tokens (``+``-joined when multiple
    judges), aligned with ``judge.py`` batch folder naming. The ``p_*`` segment lists
    user/persona models inferred from every source's sibling ``p_*`` folder (sorted,
    ``+``-joined when more than one). Provider agent, turn count, and runs-per-prompt
    reuse the first source's generation folder when parsable; otherwise falls back to
    ``t30__r1`` and ``a_unknown``. A timestamp suffix keeps same-day batches distinct.

    Args:
        source_eval_dirs: Non-empty list of resolved evaluation directories.
        combined: Merged results dataframe (used for judge slug inference).
        judge_slug: Optional override for the judge segment.

    Returns:
        Directory basename (no path separators), e.g.
        ``j_gpt-5.4x1__p_claude_opus_4_5+gpt_5_2__a_gpt_5_4_nano__t30__r1__20260422_153000``.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    judge = _judge_slug_for_pool(combined, source_eval_dirs, judge_slug=judge_slug)
    persona_slug = _persona_slug_for_pool(source_eval_dirs)
    gen = _generation_folder_for_eval(source_eval_dirs[0])
    if gen is not None:
        try:
            meta = parse_generation_run_folder_name(gen.name)
            agent = meta["agent"]
            t = meta["turns"]
            r = meta["runs"]
            return f"j_{judge}__p_{persona_slug}__a_{agent}__t{t}__r{r}__{ts}"
        except ValueError:
            pass
    return f"j_{judge}__p_{persona_slug}__a_unknown__t30__r1__{ts}"


def _annotate_pooled_results(
    results: dict[str, Any],
    combined: Any,
    source_eval_dirs: list[Path],
    rows_per_source: list[int],
) -> dict[str, Any]:
    """
    Return *results* shallow-merged with pooled reporting and provenance.

    Sets top-level ``judge_model`` and ``persona_model`` to the literal ``"pooled"``
    (so charts and JSON match user expectations) and adds a ``pooled`` object with
    source paths, per-source row counts, total rows, and distinct ``judge_model``
    values present in the merged dataframe.

    Args:
        results: Score dict returned by ``score_results``.
        combined: Merged judge results dataframe (pandas ``DataFrame``).
        source_eval_dirs: Evaluation directories that were concatenated, in order.
        rows_per_source: Row count from each source, same order as *source_eval_dirs*.

    Returns:
        New dict: ``{**results, ...pooled fields...}`` (nested values are shared).
    """
    judges = sorted(
        {str(x) for x in combined["judge_model"].dropna().astype(str).unique()}
    )
    return {
        **results,
        "judge_model": "pooled",
        "persona_model": "pooled",
        "pooled": {
            "source_evaluation_directories": [str(p) for p in source_eval_dirs],
            "rows_per_source": rows_per_source,
            "total_rows": int(len(combined)),
            "unique_judge_models_in_data": judges,
        },
    }


def run_pooling(
    *,
    evaluations: list[str | Path],
    output_parent: str | Path,
    personas_tsv: str | Path | None,
    skip_risk_analysis: bool,
    judge_slug: str | None = None,
) -> Path:
    """
    Merge several judge runs into one ``results.csv`` and compute VERA artifacts.

    Loads each evaluation via ``ensure_results_csv`` (rebuilding from TSVs if needed),
    concatenates rows, then writes merged ``results.csv``, ``pool_metadata.json``, and
    scoring artifacts under a new synthetic ``j_<judge>__p_...`` directory.

    Args:
        evaluations: Paths to ``j_*`` folders or to ``results.csv`` inside them.
        output_parent: Directory under which the new merged ``j_*`` folder is
            created; created if missing.
        personas_tsv: Personas file for risk-level analysis, or None to skip that
            step, as in `score.run.run_scoring`.
        skip_risk_analysis: When True, skip ``score_results_by_risk`` and risk charts.
        judge_slug: Optional override for the judge name (e.g. ``gpt-4ox1+sonnet45x1``).

    Returns:
        Path to the synthetic evaluation folder (``results.csv`` and ``scores/``).

    Raises:
        FileNotFoundError: If a source path does not exist or is not a directory after
            resolving ``results.csv`` to its parent.
        ValueError: If the merged dataframe has no rows.
    """
    eval_dirs = [_resolve_eval_input(Path(p)) for p in evaluations]
    for d in eval_dirs:
        if not d.is_dir():
            raise FileNotFoundError(d)

    dfs: list[pd.DataFrame] = []
    rows_per_source: list[int] = []
    for d in eval_dirs:
        df = ensure_results_csv(d)
        rows_per_source.append(len(df))
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    if len(combined) == 0:
        raise ValueError("Combined results dataframe is empty.")

    missing_gen = [d for d in eval_dirs if _generation_folder_for_eval(d) is None]
    if missing_gen:
        preview = "; ".join(str(p) for p in missing_gen[:3])
        if len(missing_gen) > 3:
            preview += f"; ... and {len(missing_gen) - 3} more"
        print(
            "Warning: some evaluation paths are not under .../p_*/evaluations/j_* "
            f"({preview}). "
            "Merged results and scores are unchanged; the new merged j_* folder name "
            "may use unknown placeholders for persona/agent/turns/runs. "
            "Use nested paths from vera generate / vera judge for descriptive "
            "names.",
            file=sys.stderr,
        )

    synth_name = _synthetic_pooled_folder_basename(
        eval_dirs, combined, judge_slug=judge_slug
    )
    out_eval = (Path(output_parent) / synth_name).resolve()
    out_eval.mkdir(parents=True, exist_ok=True)
    results_csv = out_eval / "results.csv"
    combined.to_csv(results_csv, index=False)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_evaluation_directories": [str(p) for p in eval_dirs],
        "rows_per_source": rows_per_source,
        "pooled_results_csv": str(results_csv),
    }
    (out_eval / "pool_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    scores_json = out_eval / "scores" / "scores.json"
    results = score_results(str(results_csv), output_path=str(scores_json))
    results = _annotate_pooled_results(results, combined, eval_dirs, rows_per_source)
    save_results_json(results, str(results_csv), output_path=str(scores_json))

    print_scores(results)

    viz_path = out_eval / "scores" / "scores_visualization.png"
    try:
        create_visualizations(results, viz_path)
    except Exception as e:
        print(f"Warning: could not create standard visualizations: {e}")

    if not skip_risk_analysis:
        if personas_tsv is None:
            print("No personas file given; skipping risk-level analysis.")
        elif not Path(personas_tsv).is_file():
            print(
                f"Warning: personas TSV not found ({personas_tsv}), "
                "skipping risk analysis."
            )
        else:
            try:
                risk_results = score_results_by_risk(
                    str(results_csv), str(personas_tsv), write_json=False
                )
                risk_results["judge_model"] = "pooled"
                risk_results["persona_model"] = "pooled"
                risk_json = out_eval / "scores" / "scores_by_risk.json"
                risk_json.parent.mkdir(parents=True, exist_ok=True)
                risk_json.write_text(
                    json.dumps(risk_results, indent=2), encoding="utf-8"
                )
                risk_viz = out_eval / "scores" / "scores_by_risk_visualization.png"
                create_risk_level_visualizations(risk_results, risk_viz)
            except Exception as e:
                print(f"Warning: could not create risk-level analysis: {e}")

    print("")
    print("Pooled outputs:")
    print(f"  {results_csv}")
    print(f"  {scores_json}")
    print(f"  {viz_path}")
    risk_json = out_eval / "scores" / "scores_by_risk.json"
    if risk_json.is_file():
        print(f"  {risk_json}")
        print(f"  {out_eval / 'scores' / 'scores_by_risk_visualization.png'}")
    print(f"  {out_eval / 'pool_metadata.json'}")
    return out_eval
