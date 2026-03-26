#!/usr/bin/env python3
"""
Analyze VERA score variability across different sampling strategies.

This script loads concatenated_heor_paper2_results.csv and calculates VERA scores
for different combinations of:
- P: number of personas (10, 20, 50, 100) with stratified sampling by risk and disclosure
- R: number of conversations per persona (1, 2, 3, 4)
- T: number of max_turns per conversation (20 vs. 100)
- U: persona/user agent (GPT 5.2 vs. Opus 4.5)
- J: judge agent (GPT-4o vs. Sonnet 4.5)

For each sampling strategy, it calculates mean and standard deviation of VERA scores.

Usage:
    python scripts/analyze_score_variability_heor_paper2.py
    python scripts/analyze_score_variability_heor_paper2.py --p-values 10 20 50
    python scripts/analyze_score_variability_heor_paper2.py -o results/variability_analysis.csv
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Use non-interactive backend

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from judge.score_utils import (
    BG_COLOR,
    DIMENSIONS,
    TEXT_COLOR,
    calculate_scores_from_df,
)

# ── Model display-name mapping ─────────────────────────────────────────────────
_MODEL_DISPLAY_NAMES: dict[str, str] = {
    "gemini 3 1": "Gemini 3.1 Pro Preview",
    "gpt 5 2": "GPT 5.2",
    "grok 4": "Grok 4",
    "opus 4 6": "Opus 4.6",
    "opus 4 5": "Opus 4.5",
    # judge models
    "gpt-4o": "GPT-4o",
    "claude-sonnet-4-5-20250929": "Sonnet 4.5",
}


def display_name(raw: str) -> str:
    """Return a human-readable model name for a raw internal model string."""
    key = raw.strip().lower()
    if key in _MODEL_DISPLAY_NAMES:
        return _MODEL_DISPLAY_NAMES[key]
    # Partial-match fallback — longest key that appears in raw wins
    best = max(
        (_MODEL_DISPLAY_NAMES[k] for k in _MODEL_DISPLAY_NAMES if k in key),
        key=lambda v: len(v),
        default=None,
    )
    return best if best else raw


def load_concatenated_results(
    csv_path: Path = Path("concatenated_heor_paper2_results.csv"),
) -> pd.DataFrame:
    """
    Load the concatenated results CSV file.

    Args:
        csv_path: Path to concatenated_heor_paper2_results.csv

    Returns:
        DataFrame with all evaluation results
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Concatenated results file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows from {csv_path}")

    return df


def load_persona_metadata(personas_tsv_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load persona metadata including risk and disclosure levels.

    Args:
        personas_tsv_path: Path to personas.tsv

    Returns:
        Dict mapping persona name to metadata dict with 'risk_level' and 'disclosure_level'
    """
    df = pd.read_csv(personas_tsv_path, sep="\t", keep_default_na=False)

    metadata = {}
    for _, row in df.iterrows():
        name = str(row["Name"]).strip()
        risk_col = row.get("Short Current Suicide Risk Level") or row.get(
            "Current Suicide Risk Level", "Unknown"
        )
        risk_level = str(risk_col).strip()
        disclosure_level = str(row.get("Disclosure of Suicide Risk", "Unknown")).strip()

        # Normalize disclosure levels
        if "Not applicable" in disclosure_level or disclosure_level == "Not applicable":
            disclosure_level = "Not applicable"
        elif "High" in disclosure_level:
            disclosure_level = "High"
        elif "Moderate" in disclosure_level:
            disclosure_level = "Moderate"
        elif "Low" in disclosure_level:
            disclosure_level = "Low"
        else:
            disclosure_level = "Unknown"

        metadata[name] = {
            "risk_level": risk_level,
            "disclosure_level": disclosure_level,
        }

    return metadata


def build_strata(
    personas: List[str],
    persona_metadata: Dict[str, Dict[str, str]],
) -> Dict[Tuple[str, str], List[str]]:
    """Group personas into strata by (risk_level, disclosure_level)."""
    stratified: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for persona in personas:
        meta = persona_metadata.get(persona, {})
        risk = meta.get("risk_level", "Unknown")
        disclosure = meta.get("disclosure_level", "Unknown")
        stratified[(risk, disclosure)].append(persona)
    return stratified


def stratify_personas_by_risk_disclosure(
    personas: List[str],
    persona_metadata: Dict[str, Dict[str, str]],
    n_personas: int,
    random_seed: Optional[int] = 42,
    strata: Optional[Dict[Tuple[str, str], List[str]]] = None,
) -> List[str]:
    """
    Stratified sampling of personas by risk and disclosure levels.

    Args:
        personas: List of persona names
        persona_metadata: Dict mapping persona name to metadata
        n_personas: Number of personas to sample
        random_seed: Random seed for reproducibility
        strata: Pre-computed strata dict (pass to avoid recomputing across calls)

    Returns:
        List of sampled persona names
    """
    if random_seed is not None:
        random.seed(random_seed)

    if strata is None:
        strata = build_strata(personas, persona_metadata)

    # Calculate proportional sampling
    total_personas = len(personas)
    sampled = []

    for (risk, disclosure), persona_list in strata.items():
        # Calculate how many to sample from this stratum
        proportion = len(persona_list) / total_personas
        n_from_stratum = max(1, int(round(n_personas * proportion)))

        # Sample from this stratum
        if len(persona_list) <= n_from_stratum:
            sampled.extend(persona_list)
        else:
            sampled.extend(random.sample(persona_list, n_from_stratum))

    # If we need more, randomly sample from remaining
    if len(sampled) < n_personas:
        remaining = [p for p in personas if p not in sampled]
        needed = n_personas - len(sampled)
        if len(remaining) >= needed:
            sampled.extend(random.sample(remaining, needed))
        else:
            sampled.extend(remaining)

    # Trim to exact number if needed
    if len(sampled) > n_personas:
        sampled = sampled[:n_personas]

    return sampled


def group_rows_by_conversation(
    df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Group DataFrame rows by conversation (same filename).

    Args:
        df: DataFrame with evaluation results (must have 'filename' column)

    Returns:
        Dict mapping conversation_key (filename) to DataFrame with rows for that conversation
    """
    conversations: Dict[str, pd.DataFrame] = {}

    # Group by filename (which represents the conversation)
    if "filename" in df.columns:
        for filename, group_df_raw in df.groupby("filename"):
            # Ensure it's a DataFrame
            if isinstance(group_df_raw, pd.Series):
                continue
            group_df: pd.DataFrame = group_df_raw
            conversations[str(filename)] = group_df

    return conversations


def partition_conversations(
    conversations_by_persona: Dict[str, Dict[str, pd.DataFrame]],
    r_conversations: int,
    j_iterations: int,
    random_seed: Optional[int] = 42,
) -> List[pd.DataFrame]:
    """
    Partition conversations into non-overlapping samples.

    Args:
        conversations_by_persona: Dict mapping persona -> dict of conversation DataFrames
        r_conversations: Number of conversations per sample per persona
        j_iterations: Number of judge iterations to include per conversation
        random_seed: Random seed for reproducibility

    Returns:
        List of sample DataFrames, each containing rows for one sample
    """
    if random_seed is not None:
        random.seed(random_seed)

    # First, organize conversations by persona
    persona_conv_keys: Dict[str, List[str]] = {}
    min_num_conv_samples = float("inf")
    available_judge_iterations = float("inf")

    for persona, conversations in conversations_by_persona.items():
        conv_keys = list(conversations.keys())

        if len(conv_keys) < r_conversations:
            raise ValueError(
                f"Not enough conversations for persona '{persona}': "
                f"have {len(conv_keys)}, need {r_conversations}"
            )

        # Check judge iterations for all conversations
        for conv_key, conv_df in conversations.items():
            num_judges = len(conv_df)
            if num_judges < j_iterations:
                raise ValueError(
                    f"Not enough judge iterations for conversation '{conv_key}': "
                    f"have {num_judges}, need {j_iterations}"
                )
            # Track actual available iterations
            available_judge_iterations = min(available_judge_iterations, num_judges)

        # Shuffle conversation keys
        random.shuffle(conv_keys)
        persona_conv_keys[persona] = conv_keys

        # Calculate how many complete conversation samples we can make
        num_possible_conv_samples = len(conv_keys) // r_conversations
        min_num_conv_samples = min(min_num_conv_samples, num_possible_conv_samples)

    if min_num_conv_samples == 0 or min_num_conv_samples == float("inf"):
        raise ValueError("Cannot create any complete samples with the given parameters")

    num_conv_samples = int(min_num_conv_samples)

    # Calculate judge partitions
    # If J divides evenly into available iterations, partition without replacement
    # Otherwise, sample with replacement
    can_partition_judges = (
        available_judge_iterations != float("inf")
        and available_judge_iterations % j_iterations == 0
    )
    num_judge_partitions = (
        int(available_judge_iterations // j_iterations) if can_partition_judges else 1
    )

    # Create the partitioned samples
    sample_dfs: List[pd.DataFrame] = []

    for conv_sample_idx in range(num_conv_samples):
        for judge_partition_idx in range(num_judge_partitions):
            sample_rows = []

            for persona, conv_keys in persona_conv_keys.items():
                # Get the slice of conversations for this sample
                start_idx = conv_sample_idx * r_conversations
                end_idx = start_idx + r_conversations
                sample_conv_keys = conv_keys[start_idx:end_idx]

                for conv_key in sample_conv_keys:
                    conv_df = conversations_by_persona[persona][conv_key]

                    if can_partition_judges:
                        # Partition judges without replacement
                        j_start = judge_partition_idx * j_iterations
                        j_end = j_start + j_iterations
                        sample_rows.append(conv_df.iloc[j_start:j_end])
                    else:
                        # Sample judges WITH replacement
                        sampled_indices = random.choices(
                            range(len(conv_df)), k=j_iterations
                        )
                        sample_rows.append(conv_df.iloc[sampled_indices])

            if sample_rows:
                sample_df = pd.concat(sample_rows, ignore_index=True)
                sample_dfs.append(sample_df)

    return sample_dfs


def calculate_sample_scores(
    sample_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calculate VERA scores for a sample DataFrame.

    Args:
        sample_df: DataFrame with evaluation results

    Returns:
        Dict with overall_score and dimension_scores
    """
    if sample_df.empty:
        return {"overall_score": 0.0, "dimension_scores": {}}

    # Calculate scores
    return calculate_scores_from_df(sample_df)


def run_sampling_analysis(
    df: pd.DataFrame,
    persona_metadata: Dict[str, Dict[str, str]],
    p_values: List[int],
    r_values: List[int],
    t_values: List[int],
    u_values: List[str],
    j_values: List[str],
    random_seed: int = 42,
    n_bootstrap: int = 200,
) -> pd.DataFrame:
    """
    Run sampling analysis for all parameter combinations.

    Args:
        df: DataFrame with all evaluation results from concatenated CSV
        persona_metadata: Dict mapping persona name to metadata
        p_values: List of P values (number of personas)
        r_values: List of R values (conversations per persona)
        t_values: List of T values (max_turns)
        u_values: List of U values (persona/user models)
        j_values: List of J values (judge models)
        random_seed: Random seed for reproducibility
        n_bootstrap: Number of bootstrap iterations for std estimation

    Returns:
        DataFrame with results
    """

    def _analyze_group(
        group_df: pd.DataFrame,
        persona_model_label: str,
        provider_model_label: str,
        r_values_override: List[int] | None = None,
    ) -> List[Dict[str, Any]]:
        """Run the variability analysis for one (user, provider) group.

        Args:
            r_values_override: If provided, use these R values instead of the
                               outer r_values. Pass a capped list for per-user
                               passes where only 4 conversations per persona exist.
        """
        group_results: List[Dict[str, Any]] = []

        if "persona_name" not in group_df.columns:
            print("   ⚠️  No persona_name column, skipping")
            return group_results

        personas = group_df["persona_name"].unique()
        print(f"   Available personas: {len(personas)}")

        persona_strata = build_strata(list(personas), persona_metadata)

        for p in p_values:
            if p > len(personas):
                print(f"   Skipping P={p}: only {len(personas)} personas available")
                continue

            sampled_personas = stratify_personas_by_risk_disclosure(
                list(personas),
                persona_metadata,
                p,
                random_seed=random_seed,
                strata=persona_strata,
            )

            persona_df_raw = group_df[group_df["persona_name"].isin(sampled_personas)]
            assert isinstance(persona_df_raw, pd.DataFrame)
            persona_df = persona_df_raw

            conversations_by_persona: Dict[str, Dict[str, pd.DataFrame]] = {}
            for persona in sampled_personas:
                persona_conv_df = persona_df[
                    persona_df["persona_name"] == persona
                ].copy()
                if (
                    isinstance(persona_conv_df, pd.DataFrame)
                    and not persona_conv_df.empty
                ):
                    convs = group_rows_by_conversation(persona_conv_df)
                    if convs:
                        conversations_by_persona[str(persona)] = convs

            if not conversations_by_persona:
                continue

            active_r_values = (
                r_values_override if r_values_override is not None else r_values
            )
            for r in active_r_values:
                print(
                    f"   Sampling P={p}, R={r} "
                    f"(bootstrap n={n_bootstrap} × 2 types)...",
                    end=" ",
                )
                # Separate deterministic seeds per (p, r) and bootstrap type
                # so the two types are independent but reproducible.
                # rng_conv = np.random.default_rng(random_seed + p * 1000 + r)  # conversation bootstrap (commented out)
                rng_pers = np.random.default_rng(random_seed + p * 1000 + r + 500_000)

                persona_names = list(conversations_by_persona.keys())

                for boot_type, rng in [
                    # ("conversation", rng_conv),  # conversation bootstrap — commented out, re-enable to compare
                    ("persona", rng_pers),
                ]:
                    sample_scores: List[float] = []
                    sample_dim_scores: Dict[str, List[float]] = defaultdict(list)

                    for _ in range(n_bootstrap):
                        boot_rows: List[pd.DataFrame] = []

                        # ── Conversation bootstrap (commented out) ──────────────
                        # if boot_type == "conversation":
                        #     # Fix the P stratified personas; resample R
                        #     # conversations per persona WITH replacement
                        #     # (standard bootstrap). replace=False would
                        #     # degenerate to std=0 when R=N.
                        #     for persona_convs in conversations_by_persona.values():
                        #         conv_list = list(persona_convs.values())
                        #         chosen = rng.choice(
                        #             len(conv_list), size=r, replace=True
                        #         )
                        #         for ci in chosen:
                        #             boot_rows.append(conv_list[int(ci)])
                        # else:
                        if True:
                            # Persona bootstrap: resample P personas WITH
                            # replacement from the already-stratified persona
                            # pool, then take R conversations per draw WITHOUT
                            # replacement.
                            chosen_pi = rng.choice(
                                len(persona_names),
                                size=len(persona_names),
                                replace=True,
                            )
                            for pi in chosen_pi:
                                persona_convs = conversations_by_persona[
                                    persona_names[int(pi)]
                                ]
                                conv_list = list(persona_convs.values())
                                r_actual = min(r, len(conv_list))
                                chosen_ci = rng.choice(
                                    len(conv_list), size=r_actual, replace=False
                                )
                                for ci in chosen_ci:
                                    boot_rows.append(conv_list[int(ci)])

                        if not boot_rows:
                            continue
                        sample_df = pd.concat(boot_rows, ignore_index=True)
                        scores = calculate_sample_scores(sample_df)
                        sample_scores.append(scores["overall_score"])
                        for dim, dim_data in scores.get("dimension_scores", {}).items():
                            sample_dim_scores[dim].append(
                                dim_data.get("vera_score", 0.0)
                            )

                    num_samples = len(sample_scores)
                    mean_score = (
                        sum(sample_scores) / num_samples if sample_scores else 0.0
                    )
                    std_score = (
                        (
                            sum((s - mean_score) ** 2 for s in sample_scores)
                            / num_samples
                        )
                        ** 0.5
                        if sample_scores
                        else 0.0
                    )
                    min_score = min(sample_scores) if sample_scores else 0.0
                    max_score = max(sample_scores) if sample_scores else 0.0
                    # Percentile-based 95% CI (non-parametric, no normality assumption)
                    sorted_scores = sorted(sample_scores)
                    n_s = len(sorted_scores)
                    ci_lo = sorted_scores[int(0.025 * n_s)] if n_s > 0 else 0.0
                    ci_hi = (
                        sorted_scores[min(int(0.975 * n_s), n_s - 1)]
                        if n_s > 0
                        else 0.0
                    )

                    row: Dict[str, Any] = {
                        "bootstrap_type": boot_type,
                        "persona_model": persona_model_label,
                        "provider_model": provider_model_label,
                        "P": p,
                        "R": r,
                        "num_samples": num_samples,
                        "overall_mean": round(mean_score, 2),
                        "overall_std": round(std_score, 2),
                        "overall_min": round(min_score, 2),
                        "overall_max": round(max_score, 2),
                        "overall_ci_lower": round(ci_lo, 2),
                        "overall_ci_upper": round(ci_hi, 2),
                    }
                    for dim in DIMENSIONS:
                        if dim in sample_dim_scores and sample_dim_scores[dim]:
                            dim_list = sample_dim_scores[dim]
                            dm = sum(dim_list) / len(dim_list)
                            ds = (
                                sum((s - dm) ** 2 for s in dim_list) / len(dim_list)
                            ) ** 0.5
                            sorted_dim = sorted(dim_list)
                            nd = len(sorted_dim)
                            row[f"{dim}_mean"] = round(dm, 2)
                            row[f"{dim}_std"] = round(ds, 2)
                            row[f"{dim}_min"] = round(min(dim_list), 2)
                            row[f"{dim}_max"] = round(max(dim_list), 2)
                            row[f"{dim}_ci_lower"] = round(
                                sorted_dim[int(0.025 * nd)], 2
                            )
                            row[f"{dim}_ci_upper"] = round(
                                sorted_dim[min(int(0.975 * nd), nd - 1)], 2
                            )
                    group_results.append(row)

                print("done")

        return group_results

    # ── main analysis ─────────────────────────────────────────────────────────
    results: List[Dict[str, Any]] = []

    # Limit to max_turns=20 only
    if "max_turns" in df.columns:
        df_filtered = df[df["max_turns"] == 20].copy()
        assert isinstance(df_filtered, pd.DataFrame)
        df = df_filtered
        print(f"   Filtered to max_turns=20: {len(df)} rows")

    # Group by user_llm and provider_llm — judge models are pooled within each group
    groupby_cols = ["user_llm", "provider_llm"]
    groupby_cols = [col for col in groupby_cols if col in df.columns]

    print("\n🔍 Available values in data:")
    for col in groupby_cols:
        print(f"   {col}: {list(df[col].unique())[:10]}")
    print("\n🔍 Filtering for:")
    print(f"   user_llm: {u_values}")
    print(f"   judge models pooled (both included): {j_values}")

    # Pass 1: per-user-agent analysis
    for group_key, group_df_raw in df.groupby(groupby_cols, dropna=False):
        if isinstance(group_df_raw, pd.Series):
            continue
        group_df: pd.DataFrame = group_df_raw

        if isinstance(group_key, tuple):
            user_llm = str(group_key[0]) if len(group_key) > 0 else ""
            provider_llm = str(group_key[1]) if len(group_key) > 1 else ""
        else:
            user_llm = str(group_key)
            provider_llm = ""

        if u_values and user_llm not in u_values:
            continue

        n_judges = (
            group_df["judge_model"].nunique()
            if "judge_model" in group_df.columns
            else "?"
        )
        print(
            f"\n📊 Analyzing: U={user_llm}, Provider={provider_llm} "
            f"({n_judges} judge models pooled)"
        )

        # Per-user-agent pass: cap R at 4 (only 4 conversations per user per persona)
        r_per_user = [r for r in r_values if r <= 4]
        results.extend(
            _analyze_group(
                group_df, user_llm, provider_llm, r_values_override=r_per_user
            )
        )

    # Pass 2: pooled analysis — both user agents combined per provider
    if "provider_llm" in df.columns:
        print("\n📊 Pass 2: pooled user agents per provider")
        for provider_llm, pool_df_raw in df.groupby("provider_llm", dropna=False):
            if isinstance(pool_df_raw, pd.Series):
                continue
            pool_df: pd.DataFrame = pool_df_raw
            provider_llm_str = str(provider_llm)

            # Respect u_values filter: only pool the requested user agents
            if u_values and "user_llm" in pool_df.columns:
                pool_df_filtered = pool_df[pool_df["user_llm"].isin(u_values)]
                assert isinstance(pool_df_filtered, pd.DataFrame)
                pool_df = pool_df_filtered

            if pool_df.empty:
                continue

            print(f"\n📊 Analyzing (pooled): Provider={provider_llm_str}")
            results.extend(_analyze_group(pool_df, "pooled", provider_llm_str))

    # Pass 2b: per-judge, both user agents pooled per provider
    # Produces rows with persona_model = "{judge_model}" so plots can use
    # clean per-judge stats without needing to combine bootstrap distributions.
    if all(c in df.columns for c in ["judge_model", "provider_llm"]):
        print("\n📊 Pass 2b: per judge (users pooled) per provider")
        p2b_cols = ["judge_model", "provider_llm"]
        for group_key_2b, group_df_2b_raw in df.groupby(p2b_cols, dropna=False):
            if isinstance(group_df_2b_raw, pd.Series):
                continue
            group_df_2b: pd.DataFrame = group_df_2b_raw
            gk2b = (
                list(group_key_2b)
                if isinstance(group_key_2b, tuple)
                else [str(group_key_2b), ""]
            )
            judge_2b = str(gk2b[0]) if len(gk2b) > 0 else ""
            provider_2b = str(gk2b[1]) if len(gk2b) > 1 else ""
            if j_values and judge_2b not in j_values:
                continue
            # Respect u_values filter
            if u_values and "user_llm" in group_df_2b.columns:
                group_df_2b_f = group_df_2b[group_df_2b["user_llm"].isin(u_values)]
                assert isinstance(group_df_2b_f, pd.DataFrame)
                group_df_2b = group_df_2b_f
            if group_df_2b.empty:
                continue
            print(
                f"\n📊 Analyzing (judge pooled-users): J={judge_2b}, Provider={provider_2b}"
            )
            results.extend(_analyze_group(group_df_2b, judge_2b, provider_2b))

    # Pass 3: per (user_llm × judge_model) per provider
    # Enables the std-threshold plot to break out each of the 4 combinations.
    if all(c in df.columns for c in ["user_llm", "judge_model", "provider_llm"]):
        print("\n📊 Pass 3: per (user × judge) combinations per provider")
        p3_cols = ["user_llm", "judge_model", "provider_llm"]
        for group_key_p3, group_df_p3_raw in df.groupby(p3_cols, dropna=False):
            if isinstance(group_df_p3_raw, pd.Series):
                continue
            group_df_p3: pd.DataFrame = group_df_p3_raw
            gk = (
                list(group_key_p3)
                if isinstance(group_key_p3, tuple)
                else [str(group_key_p3), "", ""]
            )
            user_llm_p3 = str(gk[0]) if len(gk) > 0 else ""
            judge_p3 = str(gk[1]) if len(gk) > 1 else ""
            provider_p3 = str(gk[2]) if len(gk) > 2 else ""
            if u_values and user_llm_p3 not in u_values:
                continue
            if j_values and judge_p3 not in j_values:
                continue
            combo_label = f"{user_llm_p3} / {judge_p3}"
            print(
                f"\n📊 Analyzing (user×judge): U={user_llm_p3}, "
                f"J={judge_p3}, Provider={provider_p3}"
            )
            # Per-user-agent pass: cap R at 4
            r_per_user_p3 = [r for r in r_values if r <= 4]
            results.extend(
                _analyze_group(
                    group_df_p3,
                    combo_label,
                    provider_p3,
                    r_values_override=r_per_user_p3,
                )
            )

    return pd.DataFrame(results)


def plot_std_threshold(
    results_df: pd.DataFrame,
    output_path: Path,
    p_target: int = 100,
    std_threshold: float = 2.0,
    title: str | None = None,
    bootstrap_type: str = "persona",
) -> None:
    """
    At a fixed P (number of personas), show how std changes with R for each
    provider, and mark where each provider first crosses below std_threshold.

    Answers: "What is the minimum R to achieve std < threshold at P=p_target,
    for each provider?"
    """
    if results_df.empty:
        print("⚠️  No data to plot")
        return

    # Filter to the requested bootstrap type if the column exists
    df_bt = results_df
    if "bootstrap_type" in results_df.columns:
        df_bt_raw = results_df[results_df["bootstrap_type"] == bootstrap_type]
        assert isinstance(df_bt_raw, pd.DataFrame)
        df_bt = df_bt_raw
        if df_bt.empty:
            print(f"⚠️  No rows for bootstrap_type='{bootstrap_type}'")
            return

    at_p_raw = df_bt[df_bt["P"] == p_target]
    assert isinstance(at_p_raw, pd.DataFrame)
    at_p: pd.DataFrame = at_p_raw
    if at_p.empty:
        # Fall back to largest available P
        p_target = int(df_bt["P"].max())
        at_p_raw2 = df_bt[df_bt["P"] == p_target]
        assert isinstance(at_p_raw2, pd.DataFrame)
        at_p = at_p_raw2
        print(f"   ⚠️  P={p_target} not found; using P={p_target}")

    providers = sorted(at_p["provider_model"].unique())
    r_values = sorted(at_p["R"].unique())

    # Provider colour palette (consistent with other plots)
    prov_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    prov_colors = {
        p: prov_palette[i % len(prov_palette)] for i, p in enumerate(providers)
    }

    # ── identify persona_model categories in the data ─────────────────────────
    all_persona_models = sorted(at_p["persona_model"].unique())
    # Combo entries were added by Pass 3: "user_llm / judge_model"
    combo_models = sorted([str(m) for m in all_persona_models if " / " in str(m)])
    solo_user_models = [
        m for m in all_persona_models if m != "pooled" and " / " not in str(m)
    ]
    use_combos = len(combo_models) > 0

    # ── colour / style for the 4 (user × judge) combos ────────────────────────
    combo_users: list[str] = sorted(set(c.split(" / ")[0] for c in combo_models))
    combo_judges: list[str] = sorted(set(c.split(" / ")[1] for c in combo_models))

    # Two shades per user family; judge distinguished by line style
    _user_palettes: list[tuple[str, str]] = [
        ("#6baed6", "#08519c"),  # blues   — first user LLM
        ("#fd8d3c", "#a63603"),  # oranges — second user LLM
        ("#74c476", "#006d2c"),  # greens
        ("#de2d26", "#67000d"),  # reds
    ]
    user_color_pairs: dict[str, tuple[str, str]] = {
        u: _user_palettes[i % len(_user_palettes)] for i, u in enumerate(combo_users)
    }
    _judge_ls_cycle = ["--", "-."]
    judge_linestyles: dict[str, str] = {
        j: _judge_ls_cycle[i % len(_judge_ls_cycle)] for i, j in enumerate(combo_judges)
    }

    def get_combo_style(combo: str) -> dict[str, Any]:
        parts = combo.split(" / ", 1)
        user_k = parts[0] if len(parts) > 0 else ""
        judge_k = parts[1] if len(parts) > 1 else ""
        light, dark = user_color_pairs.get(user_k, ("#888888", "#444444"))
        j_idx = combo_judges.index(judge_k) if judge_k in combo_judges else 0
        return {
            "color": dark if j_idx else light,
            "linestyle": judge_linestyles.get(judge_k, "--"),
            "linewidth": 1.3,
            "alpha": 0.85,
        }

    from matplotlib.lines import Line2D

    n_providers = len(providers)
    ncols = 2
    nrows = (n_providers + 1) // 2  # ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), squeeze=False)
    fig.patch.set_facecolor(BG_COLOR)

    threshold_crossings: list[dict[str, object]] = []

    for idx, provider in enumerate(providers):
        row_idx, col_idx = divmod(idx, ncols)
        ax = axes[row_idx][col_idx]
        color = prov_colors[provider]

        prov_df_raw = at_p[at_p["provider_model"] == provider]
        assert isinstance(prov_df_raw, pd.DataFrame)
        prov_df: pd.DataFrame = prov_df_raw

        # ── pooled line (both users + both judges) ─────────────────────────
        pooled_raw = prov_df[prov_df["persona_model"] == "pooled"]
        assert isinstance(pooled_raw, pd.DataFrame)
        pooled_df = pooled_raw.sort_values("R")
        if not pooled_df.empty:
            r_arr = pooled_df["R"].to_numpy(dtype=float)
            std_arr = pooled_df["overall_std"].to_numpy(dtype=float)
            ax.plot(
                r_arr,
                std_arr,
                color=color,
                linestyle="-",
                linewidth=2.8,
                alpha=1.0,
                marker="o",
                markersize=7,
                label="pooled",
                zorder=4,
            )
            # Annotate first crossing below threshold
            crossing_r: float | None = None
            crossing_std: float | None = None
            for r_val, s_val in zip(r_arr, std_arr):
                if s_val < std_threshold:
                    crossing_r = float(r_val)
                    crossing_std = float(s_val)
                    break
            if crossing_r is not None and crossing_std is not None:
                ax.scatter(
                    [crossing_r],
                    [crossing_std],
                    color=color,
                    s=110,
                    zorder=6,
                    edgecolors="white",
                    linewidths=1.5,
                )
                ax.annotate(
                    f"R={int(crossing_r)}, std={crossing_std:.2f}",
                    xy=(crossing_r, crossing_std),
                    xytext=(crossing_r + 0.15, crossing_std + 0.08),
                    fontsize=8,
                    color=color,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.7),
                )
                threshold_crossings.append(
                    {
                        "provider": provider,
                        "min_R_pooled": int(crossing_r),
                        "std_at_min_R": round(crossing_std, 3),
                    }
                )
            else:
                ax.annotate(
                    f"never < {std_threshold}",
                    xy=(r_arr[-1], std_arr[-1]),
                    xytext=(r_arr[-1] - 0.5, std_arr[-1] + 0.1),
                    fontsize=8,
                    color=color,
                    style="italic",
                )

        # ── 4 combo lines (user × judge) ───────────────────────────────────
        if use_combos:
            for combo in combo_models:
                c_raw = prov_df[prov_df["persona_model"] == combo]
                assert isinstance(c_raw, pd.DataFrame)
                c_df = c_raw.sort_values("R")
                if c_df.empty:
                    continue
                style = get_combo_style(combo)
                parts = combo.split(" / ", 1)
                lbl = (
                    f"{display_name(parts[0])} / {display_name(parts[1])}"
                    if len(parts) == 2
                    else combo
                )
                ax.plot(
                    c_df["R"].to_numpy(dtype=float),
                    c_df["overall_std"].to_numpy(dtype=float),
                    marker="o",
                    markersize=4,
                    label=lbl,
                    zorder=3,
                    **style,
                )
        else:
            # Fallback if Pass 3 data not present: show per-user lines
            for um in solo_user_models:
                u_raw = prov_df[prov_df["persona_model"] == um]
                assert isinstance(u_raw, pd.DataFrame)
                u_df = u_raw.sort_values("R")
                if u_df.empty:
                    continue
                ax.plot(
                    u_df["R"].to_numpy(dtype=float),
                    u_df["overall_std"].to_numpy(dtype=float),
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    marker="o",
                    markersize=4,
                    label=f"user: {display_name(um)}",
                    zorder=2,
                )

        # ── threshold line ────────────────────────────────────────────────────
        ax.axhline(
            y=std_threshold,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold  std={std_threshold}",
            zorder=5,
        )

        # ── per-subplot legend ────────────────────────────────────────────────
        legend_handles: list[Line2D] = [
            Line2D(
                [0],
                [0],
                color=color,
                linestyle="-",
                linewidth=2.8,
                label="pooled (all users + judges)",
            ),
        ]
        if use_combos:
            for combo in combo_models:
                style = get_combo_style(combo)
                parts = combo.split(" / ", 1)
                lbl = (
                    f"{display_name(parts[0])} / {display_name(parts[1])}"
                    if len(parts) == 2
                    else combo
                )
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                        alpha=style["alpha"],
                        label=lbl,
                    )
                )
        else:
            for um in solo_user_models:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                        label=f"user: {display_name(um)}",
                    )
                )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=f"Threshold  std={std_threshold}",
            ),
        )
        ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

        ax.set_xlabel("R (conversations per persona)", fontsize=10, color=TEXT_COLOR)
        ax.set_ylabel(
            "Standard Deviation of VERA Score Score", fontsize=10, color=TEXT_COLOR
        )
        ax.set_xticks([int(r) for r in r_values])
        ax.set_title(
            display_name(provider), fontsize=11, fontweight="bold", color=TEXT_COLOR
        )
        ax.set_ylim(0, 3)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("white")

    # Hide any unused subplots (if odd number of providers)
    for empty_idx in range(n_providers, nrows * ncols):
        row_idx, col_idx = divmod(empty_idx, ncols)
        axes[row_idx][col_idx].set_visible(False)

    fig.suptitle(
        title
        or f"Minimum R to Achieve Std < {std_threshold} for each Provder LLM (P={p_target})",
        fontsize=13,
        fontweight="bold",
        color=TEXT_COLOR,
        y=1.01,
    )

    # ── print summary table ───────────────────────────────────────────────────
    if threshold_crossings:
        print(f"\n📊 Minimum R to achieve std < {std_threshold} at P={p_target}:")
        for row in sorted(threshold_crossings, key=lambda x: x["min_R_pooled"]):  # type: ignore[arg-type]
            print(
                f"   {row['provider']:<30}  R={row['min_R_pooled']}  (std={row['std_at_min_R']})"
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"📈 Std-threshold plot saved to: {output_path}")


def plot_std_threshold_combined(
    results_df: pd.DataFrame,
    output_path: Path,
    p_target: int = 100,
    std_threshold: float = 2.0,
    title: str | None = None,
    bootstrap_type: str = "persona",
    judge_filter: str | None = None,
    use_ci_width: bool = False,
) -> None:
    """
    Single-chart version of plot_std_threshold: one line per provider,
    all on the same axes, with the legend identifying providers.

    When judge_filter is None, uses the "pooled" persona_model rows (all users
    and judges combined).  When judge_filter is a string, filters the Pass-3
    combo rows whose persona_model ends with that judge substring, then averages
    std across user agents to produce one line per provider.

    use_ci_width: If True, plot 95% CI width (ci_upper − ci_lower) on the y-axis
                  instead of standard deviation.
    """
    if results_df.empty:
        print("⚠️  No data to plot")
        return

    df_bt = results_df
    if "bootstrap_type" in results_df.columns:
        df_bt_raw = results_df[results_df["bootstrap_type"] == bootstrap_type]
        assert isinstance(df_bt_raw, pd.DataFrame)
        df_bt = df_bt_raw
        if df_bt.empty:
            print(f"⚠️  No rows for bootstrap_type='{bootstrap_type}'")
            return

    at_p_raw = df_bt[df_bt["P"] == p_target]
    assert isinstance(at_p_raw, pd.DataFrame)
    at_p: pd.DataFrame = at_p_raw
    if at_p.empty:
        p_target = int(df_bt["P"].max())
        at_p_raw2 = df_bt[df_bt["P"] == p_target]
        assert isinstance(at_p_raw2, pd.DataFrame)
        at_p = at_p_raw2

    # Select rows by exact persona_model key:
    #   judge_filter=None  → "pooled" (both users + both judges, from Pass 2)
    #   judge_filter=<str> → the raw judge model name (both users, one judge,
    #                         from Pass 2b — bootstrapped directly, no combining)
    persona_model_key = "pooled" if judge_filter is None else judge_filter
    subset_raw = at_p[at_p["persona_model"] == persona_model_key]
    assert isinstance(subset_raw, pd.DataFrame)
    plot_df = subset_raw.copy()
    if plot_df.empty:
        print(f"⚠️  No rows with persona_model='{persona_model_key}'")
        return

    # Compute CI width if needed
    if use_ci_width:
        plot_df = plot_df.copy()
        plot_df["ci_width"] = plot_df["overall_ci_upper"] - plot_df["overall_ci_lower"]

    y_col = "ci_width" if use_ci_width else "overall_std"
    y_label = (
        "95% CI width (bootstrap)"
        if use_ci_width
        else "Standard Deviation of VERA Score"
    )
    y_max = 10.0 if use_ci_width else 2.5
    y_ticks = list(range(0, int(y_max) + 1, 2)) if use_ci_width else None

    def _get_y_series(prov_df: pd.DataFrame) -> pd.DataFrame:
        result = prov_df.sort_values("R")[["R", y_col]]
        assert isinstance(result, pd.DataFrame)
        return result

    providers = sorted(plot_df["provider_model"].unique())
    r_values = sorted(plot_df["R"].unique())

    prov_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    prov_colors = {
        p: prov_palette[i % len(prov_palette)] for i, p in enumerate(providers)
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor("white")

    for provider in providers:
        prov_df_raw = plot_df[plot_df["provider_model"] == provider]
        assert isinstance(prov_df_raw, pd.DataFrame)
        y_series = _get_y_series(prov_df_raw)
        if y_series.empty:
            continue
        color = prov_colors[provider]
        ax.plot(
            y_series["R"].tolist(),
            y_series[y_col].tolist(),
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=7,
            label=display_name(provider),
        )

    if not use_ci_width:
        ax.axhline(
            std_threshold,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold  std={std_threshold}",
        )
    ax.set_xlabel("R (conversations per persona)", fontsize=11, color=TEXT_COLOR)
    ax.set_ylabel(y_label, fontsize=11, color=TEXT_COLOR)
    ax.set_xticks([int(r) for r in r_values])
    ax.set_ylim(0, y_max)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    if title is None:
        judge_label = f", judge={display_name(judge_filter)}" if judge_filter else ""
        if use_ci_width:
            title = (
                "95% CI Width of VERA-MH v1 Score by Number of "
                f"Conversations per Profile (P={p_target}{judge_label})"
            )
        else:
            title = (
                "Standard Deviation of VERA-MH v1 Score by Number of "
                f"Conversations per Profile (P={p_target}{judge_label})"
            )
    fig.suptitle(title, fontsize=13, fontweight="bold", color=TEXT_COLOR)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"📈 Combined std-threshold plot saved to: {output_path}")


def plot_score_convergence_by_provider(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str = "VERA Score Range by Provider, Number of Personas, and Conversations per Persona (R)",
    bootstrap_type: str = "persona",
) -> None:
    """
    One subplot per provider.  For each P value on the x-axis, draw one box per
    R value (conversations per persona) showing the range of scores observed:

        centre line = mean score
        box          = mean ± 1 std   (spread of bootstrap samples)
        whiskers     = observed min / max across samples

    Uses the "pooled" rows (both user agents combined) so each box has a clean
    single series per R.  Boxes for different R values at the same P are offset
    side-by-side so they can be compared directly.
    """
    if results_df.empty:
        print("⚠️  No data to plot")
        return

    # Filter to requested bootstrap type if column exists
    df_bt = results_df
    if "bootstrap_type" in results_df.columns:
        df_bt_raw = results_df[results_df["bootstrap_type"] == bootstrap_type]
        assert isinstance(df_bt_raw, pd.DataFrame)
        df_bt = df_bt_raw

    # Use pooled rows only (both user agents combined)
    pooled = df_bt[df_bt["persona_model"] == "pooled"].copy()
    assert isinstance(pooled, pd.DataFrame)
    if pooled.empty:
        print("⚠️  No pooled rows found; falling back to all rows")
        pooled = df_bt.copy()

    providers = sorted(pooled["provider_model"].unique())
    r_values = sorted(pooled["R"].unique())
    p_values = sorted(pooled["P"].unique())

    # Colour per R value
    r_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    r_colors = {r: r_palette[i % len(r_palette)] for i, r in enumerate(r_values)}

    n_prov = len(providers)
    ncols = 2
    nrows = (n_prov + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows), squeeze=False)
    fig.patch.set_facecolor(BG_COLOR)

    # Fixed y-limits — wide enough for min/max labels to breathe
    y_lo = 0.0
    y_hi = 100.0

    # Side-by-side offset within each P group
    n_r = len(r_values)
    box_width = 0.6 / n_r  # fraction of gap between P positions
    offsets = np.linspace(-(n_r - 1) / 2, (n_r - 1) / 2, n_r) * box_width

    from matplotlib.lines import Line2D

    for idx, provider in enumerate(providers):
        row_idx, col_idx = divmod(idx, ncols)
        ax = axes[row_idx][col_idx]

        prov_df_raw = pooled[pooled["provider_model"] == provider]
        assert isinstance(prov_df_raw, pd.DataFrame)
        prov_df: pd.DataFrame = prov_df_raw

        # Assign integer x positions to P values
        p_positions = {p: i for i, p in enumerate(p_values)}

        for r_idx, r in enumerate(r_values):
            r_df_raw = prov_df[prov_df["R"] == r]
            assert isinstance(r_df_raw, pd.DataFrame)
            r_df: pd.DataFrame = r_df_raw.sort_values("P")
            if r_df.empty:
                continue

            color = r_colors[r]
            offset = offsets[r_idx]

            box_stats = []
            x_centers = []
            for _, row_data in r_df.iterrows():
                p_val = row_data["P"]
                mean_v = float(row_data["overall_mean"])
                std_v = float(row_data["overall_std"])
                min_v = float(row_data["overall_min"])
                max_v = float(row_data["overall_max"])

                box_stats.append(
                    {
                        "med": mean_v,
                        "q1": mean_v - std_v,
                        "q3": mean_v + std_v,
                        "whislo": min_v,
                        "whishi": max_v,
                        "mean": mean_v,
                        "fliers": [],
                    }
                )
                x_centers.append(p_positions[p_val] + offset)

            bp = ax.bxp(
                box_stats,
                positions=x_centers,
                widths=box_width * 0.85,
                patch_artist=True,
                manage_ticks=False,
                showmeans=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
                patch.set_edgecolor(color)
            for element in ("whiskers", "caps", "medians"):
                for line in bp[element]:
                    line.set_color(color)
                    line.set_linewidth(1.5 if element != "medians" else 2.0)

            # Annotate min/max only for R=1 and R=8 boxes
            if r not in (1, 8):
                continue
            y_pad_max = (y_hi - y_lo) * 0.015  # 1.5% padding above max whisker
            y_pad_min = (y_hi - y_lo) * 0.025  # 2.5% padding below min whisker
            for box_i, stats in enumerate(box_stats):
                x_c = x_centers[box_i]
                ax.text(
                    x_c,
                    stats["whislo"] - y_pad_min,
                    f"{stats['whislo']:.1f}",
                    ha="center",
                    va="top",
                    fontsize=9,
                    color=color,
                )
                ax.text(
                    x_c,
                    stats["whishi"] + y_pad_max,
                    f"{stats['whishi']:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=color,
                )

        ax.set_xticks(list(range(len(p_values))))
        ax.set_xticklabels([str(p) for p in p_values], fontsize=10)
        ax.set_xlabel("P (number of personas)", fontsize=12, color=TEXT_COLOR)
        ax.set_ylabel("VERA Overall Score", fontsize=12, color=TEXT_COLOR)
        ax.set_title(
            display_name(provider), fontsize=11, fontweight="bold", color=TEXT_COLOR
        )
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_facecolor("white")
        ax.tick_params(labelsize=10)

        # Legend: one patch per R; description suffix only on first entry
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=r_colors[r],
                markersize=10,
                label=(
                    f"R={r}  (box=mean±std, whiskers=min/max)"
                    if r_idx == 0
                    else f"R={r}"
                ),
            )
            for r_idx, r in enumerate(r_values)
        ]
        if "opus" in provider.lower():
            ax.legend(handles=legend_handles, fontsize=7, loc="lower right")

    # Hide unused subplots
    for empty_idx in range(n_prov, nrows * ncols):
        row_idx, col_idx = divmod(empty_idx, ncols)
        axes[row_idx][col_idx].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01, color=TEXT_COLOR)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"📈 Score convergence plot saved to: {output_path}")


def plot_cost_efficiency(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str = "VERA-MH Evaluation Parameters: Adding Profiles vs. Conversations",
    judge_filter: str | None = None,
    use_ci_width: bool = False,
    use_std: bool = False,
) -> None:
    """
    For each provider (2×2 subplots), show two lines:

      "Add personas"       — vary P (holding R=1, persona bootstrap)
      "Add conversations"  — vary R (holding P=min_P, persona bootstrap)

    Y-axis is score range (max − min) by default, 95% CI width if use_ci_width=True,
    or bootstrap std if use_std=True.

    Args:
        judge_filter: If given, use Pass-2b rows with persona_model == judge_filter
                      instead of the pooled rows. Pass the raw judge_model string.
        use_ci_width: If True, plot 95% CI width instead of score range.
        use_std: If True, plot bootstrap std instead of score range.
    """
    if results_df.empty or "bootstrap_type" not in results_df.columns:
        print("⚠️  No bootstrap data available for cost-efficiency plot")
        return

    # Select pooled-users rows: either all-judges ("pooled") or one judge (Pass 2b)
    pm_key = "pooled" if judge_filter is None else judge_filter
    pooled_raw = results_df[results_df["persona_model"] == pm_key]
    assert isinstance(pooled_raw, pd.DataFrame)
    pooled = pooled_raw.copy()
    if pooled.empty:
        print(f"⚠️  No rows for persona_model='{pm_key}' in cost-efficiency plot")
        return

    providers = sorted(pooled["provider_model"].unique())
    p_values = sorted(pooled["P"].unique())
    r_values = sorted(pooled["R"].unique())
    min_p = min(p_values)

    # Pre-compute score range and CI width for every row
    pooled = pooled.copy()
    pooled["score_range"] = pooled["overall_max"] - pooled["overall_min"]
    pooled["ci_width"] = pooled["overall_ci_upper"] - pooled["overall_ci_lower"]
    if use_ci_width:
        y_col = "ci_width"
        y_label = "95% CI width (bootstrap)"
        y_max = 32.0
    elif use_std:
        y_col = "overall_std"
        y_label = "Score std (bootstrap)"
        y_max = 8.0
    else:
        y_col = "score_range"
        y_label = "Score range (max − min)"
        y_max = 60.0

    # ── "add personas" trajectory: persona bootstrap, R=1, varying P ─────────
    pers_bt = pooled[
        (pooled["bootstrap_type"] == "persona") & (pooled["R"] == min(r_values))
    ].copy()
    assert isinstance(pers_bt, pd.DataFrame)

    # ── "add conversations" trajectory: persona bootstrap, P=min_p, varying R ─
    conv_bt = pooled[
        (pooled["bootstrap_type"] == "persona") & (pooled["P"] == min_p)
    ].copy()
    assert isinstance(conv_bt, pd.DataFrame)

    COLORS = [
        "#5B9BD5",
        "#ED7D31",
        "#A9D18E",
        "#FFC000",
        "#7030A0",
        "#00B0F0",
        "#FF0000",
        "#92D050",
    ]
    provider_colors = {p: COLORS[i % len(COLORS)] for i, p in enumerate(providers)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes_flat = axes.flatten()
    fig.patch.set_facecolor(BG_COLOR)

    for plot_idx, provider in enumerate(providers[:4]):
        ax = axes_flat[plot_idx]
        ax.set_facecolor("white")
        color = provider_colors[provider]

        # ── Line 1: add personas (persona bootstrap, R=1) ────────────────────
        prov_pers = pers_bt[pers_bt["provider_model"] == provider].copy()
        assert isinstance(prov_pers, pd.DataFrame)
        if not prov_pers.empty:
            prov_pers = prov_pers.sort_values("P")
            total_convs_p = (prov_pers["P"] * min(r_values)).tolist()
            ranges_p = prov_pers[y_col].tolist()
            ax.plot(
                total_convs_p,
                ranges_p,
                color=color,
                linewidth=2.5,
                linestyle="-",
                marker="o",
                markersize=7,
                label=f"Add personas (R={min(r_values)} fixed)",
            )
            # Annotate P and R values
            for x_val, y_val, p_val in zip(
                total_convs_p, ranges_p, prov_pers["P"].tolist()
            ):
                # P=10, R=1 goes to the right (shared start point with dashed line);
                # all other P=X, R=1 labels go below
                if p_val == min_p:
                    ax.annotate(
                        f"P={p_val}, R={min(r_values)}",
                        xy=(x_val, y_val),
                        xytext=(8, 0),
                        textcoords="offset points",
                        fontsize=8,
                        color="black",
                        alpha=0.85,
                        ha="left",
                        va="center",
                    )
                else:
                    ax.annotate(
                        f"P={p_val}, R={min(r_values)}",
                        xy=(x_val, y_val),
                        xytext=(0, -10),
                        textcoords="offset points",
                        fontsize=8,
                        color="black",
                        alpha=0.85,
                        ha="center",
                        va="top",
                    )

        # ── Line 2: add conversations (conversation bootstrap, P=min_p) ──────
        prov_conv = conv_bt[conv_bt["provider_model"] == provider].copy()
        assert isinstance(prov_conv, pd.DataFrame)
        if not prov_conv.empty:
            prov_conv = prov_conv.sort_values("R")
            total_convs_r = (prov_conv["P"] * prov_conv["R"]).tolist()
            ranges_r = prov_conv[y_col].tolist()
            ax.plot(
                total_convs_r,
                ranges_r,
                color=color,
                linewidth=2.5,
                linestyle="--",
                marker="s",
                markersize=7,
                label=f"Add conversations per persona (P={min_p} fixed)",
            )
            # Annotate P and R values — only R=4, 8 on the P=min_p line
            # (R=1 is already labelled by the "Add personas" solid line above)
            for x_val, y_val, r_val in zip(
                total_convs_r, ranges_r, prov_conv["R"].tolist()
            ):
                if r_val not in (4, 8):
                    continue
                ax.annotate(
                    f"P={min_p}, R={r_val}",
                    xy=(x_val, y_val),
                    xytext=(0, 12),
                    textcoords="offset points",
                    fontsize=8,
                    color="black",
                    alpha=0.85,
                    ha="center",
                    va="bottom",
                )

        ax.set_title(
            f"{display_name(provider)} as Provider agent",
            fontsize=11,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        ax.set_xlabel(
            "Total conversations evaluated (P × R)", fontsize=10, color=TEXT_COLOR
        )
        ax.set_ylabel(y_label, fontsize=10, color=TEXT_COLOR)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3, axis="both")
        ax.set_ylim(0, y_max)
        if use_ci_width:
            ax.set_yticks(list(range(0, int(y_max) + 1, 5)))
        elif use_std:
            ax.set_yticks([round(v * 0.5, 1) for v in range(0, int(y_max / 0.5) + 1)])
        ax.legend(fontsize=8, loc="upper right")

    # Hide unused subplots
    for idx in range(len(providers), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01, color=TEXT_COLOR)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"📈 Cost-efficiency plot saved to: {output_path}")


def plot_cost_efficiency_by_p(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str = "Score Range vs. Total Conversations: Fixed P, Varying R",
    judge_filter: str | None = None,
    use_std: bool = False,
    use_ci_width: bool = False,
) -> None:
    """
    For each provider (2×2 subplots), one line per P value.
    Each line's points correspond to R = 1, 2, 3, 4.
    X-axis = P × R (total conversations evaluated).
    Y-axis = score range (max − min), std, or 95% CI width, from persona bootstrap.

    Args:
        judge_filter: If given, use Pass-2b rows with persona_model == judge_filter
                      instead of the pooled rows.
        use_std: If True, plot overall_std instead of score range (max − min).
        use_ci_width: If True, plot 95% CI width (ci_upper − ci_lower) instead of score range.
    """
    if results_df.empty or "bootstrap_type" not in results_df.columns:
        print("⚠️  No bootstrap data available for cost-efficiency-by-P plot")
        return

    pm_key = "pooled" if judge_filter is None else judge_filter
    pooled_raw = results_df[results_df["persona_model"] == pm_key]
    assert isinstance(pooled_raw, pd.DataFrame)
    pooled_filt = pooled_raw[pooled_raw["bootstrap_type"] == "persona"].copy()
    assert isinstance(pooled_filt, pd.DataFrame)
    pooled_byp: pd.DataFrame = pooled_filt
    if pooled_byp.empty:
        print(f"⚠️  No persona-bootstrap rows for persona_model='{pm_key}' (by-P plot)")
        return

    pooled_byp["score_range"] = pooled_byp["overall_max"] - pooled_byp["overall_min"]
    pooled_byp["ci_width"] = (
        pooled_byp["overall_ci_upper"] - pooled_byp["overall_ci_lower"]
    )
    if use_ci_width:
        y_col = "ci_width"
        y_label = "95% CI width (bootstrap)"
        y_max = 32.0
    elif use_std:
        y_col = "overall_std"
        y_label = "Score std (bootstrap)"
        y_max = 8.0
    else:
        y_col = "score_range"
        y_label = "Score range (max − min)"
        y_max = 60.0
    providers = sorted(pooled_byp["provider_model"].dropna().unique().tolist())
    p_values = sorted(pooled_byp["P"].dropna().unique().tolist())

    LINE_STYLES = ["-", "--", "-.", ":"]
    MARKERS = ["o", "s", "^", "D", "v", "P"]
    P_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes_flat = axes.flatten()
    fig.patch.set_facecolor(BG_COLOR)

    for plot_idx, provider in enumerate(providers[:4]):
        ax = axes_flat[plot_idx]
        ax.set_facecolor("white")
        prov_df_raw = pooled_byp[pooled_byp["provider_model"] == provider]
        assert isinstance(prov_df_raw, pd.DataFrame)
        prov_df_byp: pd.DataFrame = prov_df_raw

        for p_idx, p_val in enumerate(p_values):
            p_df_raw = prov_df_byp[prov_df_byp["P"] == p_val]
            assert isinstance(p_df_raw, pd.DataFrame)
            p_df: pd.DataFrame = p_df_raw.sort_values(by=["R"])
            if p_df.empty:
                continue
            total_convs = (p_df["P"] * p_df["R"]).tolist()
            y_vals = p_df[y_col].tolist()
            r_vals = p_df["R"].tolist()
            color = P_COLORS[p_idx % len(P_COLORS)]
            ax.plot(
                total_convs,
                y_vals,
                color=color,
                linewidth=2.0,
                linestyle=LINE_STYLES[p_idx % len(LINE_STYLES)],
                marker=MARKERS[p_idx % len(MARKERS)],
                markersize=7,
                label=f"P={p_val}",
            )
            # Label R values:
            #   - P=max line: all R values, centered below each point
            #   - P=10: R=1 above the point; R=max to the right
            #   - P=25, P=50: R=1 to the left; R=max to the right
            r_max_val = max(r_vals) if r_vals else None
            is_max_p = p_val == max(p_values)
            for x_val, y_val, r_val in zip(total_convs, y_vals, r_vals):
                if is_max_p:
                    # All labels, centered below
                    ax.annotate(
                        f"R={r_val}",
                        xy=(x_val, y_val),
                        xytext=(0, -10),
                        textcoords="offset points",
                        fontsize=8,
                        color=color,
                        alpha=0.85,
                        ha="center",
                        va="top",
                    )
                elif r_val == 1:
                    # R=1 label: above for P=10, left for others
                    if p_val == min(p_values):
                        ax.annotate(
                            f"R={r_val}",
                            xy=(x_val, y_val),
                            xytext=(8, 0),
                            textcoords="offset points",
                            fontsize=8,
                            color=color,
                            alpha=0.85,
                            ha="left",
                            va="center",
                        )
                    else:
                        ax.annotate(
                            f"R={r_val}",
                            xy=(x_val, y_val),
                            xytext=(-8, 0),
                            textcoords="offset points",
                            fontsize=8,
                            color=color,
                            alpha=0.85,
                            ha="right",
                            va="center",
                        )
                elif r_val == r_max_val:
                    # R=max label: to the right
                    ax.annotate(
                        f"R={r_val}",
                        xy=(x_val, y_val),
                        xytext=(8, 0),
                        textcoords="offset points",
                        fontsize=8,
                        color=color,
                        alpha=0.85,
                        ha="left",
                        va="center",
                    )

        ax.set_title(
            display_name(provider), fontsize=11, fontweight="bold", color=TEXT_COLOR
        )
        ax.set_xlabel(
            "Total conversations evaluated (P × R)", fontsize=10, color=TEXT_COLOR
        )
        ax.set_ylabel(y_label, fontsize=10, color=TEXT_COLOR)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3, axis="both")
        ax.set_ylim(0, y_max)
        if use_ci_width:
            ax.set_yticks(list(range(0, int(y_max) + 1, 5)))
        ax.legend(fontsize=8, loc="upper right")

    for idx in range(len(providers), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01, color=TEXT_COLOR)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"📈 Cost-efficiency-by-P plot saved to: {output_path}")


def plot_cost_efficiency_by_p_combined(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str = "Score Range vs. Total Conversations: Fixed P, Varying R (All Providers)",
    judge_filter: str | None = None,
) -> None:
    """
    Single chart with all providers overlaid.
    Colour = provider model, line style = number of personas (P).
    X-axis = P × R, Y-axis = score range (max − min), persona bootstrap.

    Args:
        judge_filter: If given, use Pass-2b rows with persona_model == judge_filter
                      instead of the pooled rows.
    """
    if results_df.empty or "bootstrap_type" not in results_df.columns:
        print("⚠️  No bootstrap data available for cost-efficiency-by-P combined plot")
        return

    pm_key = "pooled" if judge_filter is None else judge_filter
    pooled_raw = results_df[results_df["persona_model"] == pm_key]
    assert isinstance(pooled_raw, pd.DataFrame)
    pooled_filt = pooled_raw[pooled_raw["bootstrap_type"] == "persona"].copy()
    assert isinstance(pooled_filt, pd.DataFrame)
    data: pd.DataFrame = pooled_filt
    if data.empty:
        print(
            f"⚠️  No persona-bootstrap rows for persona_model='{pm_key}' (combined by-P plot)"
        )
        return

    data["score_range"] = data["overall_max"] - data["overall_min"]
    providers = sorted(data["provider_model"].dropna().unique().tolist())
    p_values = sorted(data["P"].dropna().unique().tolist())

    PROV_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    MARKERS = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor("white")

    for prov_idx, provider in enumerate(providers):
        color = PROV_COLORS[prov_idx % len(PROV_COLORS)]
        prov_df_raw = data[data["provider_model"] == provider]
        assert isinstance(prov_df_raw, pd.DataFrame)
        prov_df: pd.DataFrame = prov_df_raw

        for p_idx, p_val in enumerate(p_values):
            p_df_raw = prov_df[prov_df["P"] == p_val]
            assert isinstance(p_df_raw, pd.DataFrame)
            p_df: pd.DataFrame = p_df_raw.sort_values(by=["R"])
            if p_df.empty:
                continue
            total_convs = (p_df["P"] * p_df["R"]).tolist()
            ranges = p_df["score_range"].tolist()
            ls = LINE_STYLES[p_idx % len(LINE_STYLES)]
            marker = MARKERS[p_idx % len(MARKERS)]
            # Only add label for first P value per provider (legend shows provider+P)
            ax.plot(
                total_convs,
                ranges,
                color=color,
                linewidth=2.0,
                linestyle=ls,
                marker=marker,
                markersize=6,
                label=f"{display_name(provider)}, P={p_val}",
            )

    ax.set_xlabel(
        "Total conversations evaluated (P × R)", fontsize=11, color=TEXT_COLOR
    )
    ax.set_ylabel("Score range (max − min)", fontsize=11, color=TEXT_COLOR)
    ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT_COLOR)
    ax.set_ylim(0, 60)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3, axis="both")

    # Legend: two-column, outside the plot area
    ax.legend(
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        ncol=1,
        framealpha=0.9,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"📈 Combined cost-efficiency-by-P plot saved to: {output_path}")


def plot_cost_efficiency_by_r(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str = "Score Range vs. Total Conversations: Fixed R, Varying P",
    judge_filter: str | None = None,
) -> None:
    """
    For each provider (2×2 subplots), one line per R value.
    Each line's points correspond to the available P values (e.g. 10, 50, 100).
    X-axis = P × R (total conversations evaluated).
    Y-axis = score range (max − min) from persona bootstrap.

    Args:
        judge_filter: If given, use Pass-2b rows with persona_model == judge_filter
                      instead of the pooled rows.
    """
    if results_df.empty or "bootstrap_type" not in results_df.columns:
        print("⚠️  No bootstrap data available for cost-efficiency-by-R plot")
        return

    pm_key = "pooled" if judge_filter is None else judge_filter
    pooled_raw = results_df[results_df["persona_model"] == pm_key]
    assert isinstance(pooled_raw, pd.DataFrame)
    pooled_filt2 = pooled_raw[pooled_raw["bootstrap_type"] == "persona"].copy()
    assert isinstance(pooled_filt2, pd.DataFrame)
    pooled_byr: pd.DataFrame = pooled_filt2
    if pooled_byr.empty:
        print(f"⚠️  No persona-bootstrap rows for persona_model='{pm_key}' (by-R plot)")
        return

    pooled_byr["score_range"] = pooled_byr["overall_max"] - pooled_byr["overall_min"]
    providers = sorted(pooled_byr["provider_model"].dropna().unique().tolist())
    r_values = sorted(pooled_byr["R"].dropna().unique().tolist())

    LINE_STYLES = ["-", "--", "-.", ":"]
    MARKERS = ["o", "s", "^", "D", "v", "P"]
    R_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes_flat = axes.flatten()
    fig.patch.set_facecolor(BG_COLOR)

    for plot_idx, provider in enumerate(providers[:4]):
        ax = axes_flat[plot_idx]
        ax.set_facecolor("white")
        prov_df_raw = pooled_byr[pooled_byr["provider_model"] == provider]
        assert isinstance(prov_df_raw, pd.DataFrame)
        prov_df_byr: pd.DataFrame = prov_df_raw

        for r_idx, r_val in enumerate(r_values):
            r_df_raw = prov_df_byr[prov_df_byr["R"] == r_val]
            assert isinstance(r_df_raw, pd.DataFrame)
            r_df: pd.DataFrame = r_df_raw.sort_values(by=["P"])
            if r_df.empty:
                continue
            total_convs = (r_df["P"] * r_df["R"]).tolist()
            ranges = r_df["score_range"].tolist()
            p_vals = r_df["P"].tolist()
            color = R_COLORS[r_idx % len(R_COLORS)]
            ax.plot(
                total_convs,
                ranges,
                color=color,
                linewidth=2.0,
                linestyle=LINE_STYLES[r_idx % len(LINE_STYLES)],
                marker=MARKERS[r_idx % len(MARKERS)],
                markersize=7,
                label=f"R={r_val}",
            )
            for x_val, y_val, p_val in zip(total_convs, ranges, p_vals):
                ax.annotate(
                    f"P={p_val}",
                    xy=(x_val, y_val),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                    alpha=0.85,
                )

        ax.set_title(
            display_name(provider), fontsize=11, fontweight="bold", color=TEXT_COLOR
        )
        ax.set_xlabel(
            "Total conversations evaluated (P × R)", fontsize=10, color=TEXT_COLOR
        )
        ax.set_ylabel("Score range (max − min)", fontsize=10, color=TEXT_COLOR)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3, axis="both")
        ax.set_ylim(0, 60)
        ax.legend(fontsize=8, loc="upper right")

    for idx in range(len(providers), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01, color=TEXT_COLOR)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"📈 Cost-efficiency-by-R plot saved to: {output_path}")


def plot_bootstrap_type_comparison(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str = "Bootstrap Std Comparison: Persona Resampling vs. Conversation Resampling",
) -> None:
    """
    For each provider (2x2 subplots), plot std vs R on the pooled rows for
    both bootstrap types on the same axes.

    Solid line  = persona bootstrap   (variance from which patients are sampled)
    Dashed line = conversation bootstrap (variance from which conversations per patient)

    This answers: which source of variance dominates, and do they tell the
    same story about the minimum R needed?
    """
    if results_df.empty or "bootstrap_type" not in results_df.columns:
        print("⚠️  No data with bootstrap_type column to compare")
        return

    # Restrict to pooled user-agent rows
    df = results_df[results_df["persona_model"] == "pooled"].copy()
    assert isinstance(df, pd.DataFrame)
    if df.empty:
        print("⚠️  No pooled rows found for bootstrap comparison")
        return

    providers = sorted(df["provider_model"].unique())
    r_values = sorted(df["R"].unique())
    bt_styles = {
        "persona": {"linestyle": "-", "label": "Persona resampling"},
        "conversation": {"linestyle": "--", "label": "Conversation resampling"},
    }
    bt_types = [
        bt for bt in ["persona", "conversation"] if bt in df["bootstrap_type"].unique()
    ]

    COLORS = [
        "#5B9BD5",
        "#ED7D31",
        "#A9D18E",
        "#FFC000",
        "#7030A0",
        "#00B0F0",
        "#FF0000",
        "#92D050",
    ]
    provider_colors = {p: COLORS[i % len(COLORS)] for i, p in enumerate(providers)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes_flat = axes.flatten()
    fig.patch.set_facecolor(BG_COLOR)

    STD_THRESHOLD = 1.0

    for plot_idx, provider in enumerate(providers[:4]):
        ax = axes_flat[plot_idx]
        ax.set_facecolor(BG_COLOR)
        color = provider_colors[provider]

        for bt in bt_types:
            subset_raw = df[
                (df["provider_model"] == provider) & (df["bootstrap_type"] == bt)
            ]
            assert isinstance(subset_raw, pd.DataFrame)
            subset = subset_raw.copy()
            if subset.empty:
                continue

            std_by_r = subset.groupby("R")["overall_std"].mean().reindex(r_values)

            style = bt_styles[bt]
            ax.plot(
                r_values,
                std_by_r.values,
                color=color,
                linewidth=2.2,
                linestyle=style["linestyle"],
                marker="o",
                markersize=5,
                label=style["label"],
            )

        ax.axhline(
            STD_THRESHOLD,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label=f"Threshold ({STD_THRESHOLD})",
        )
        ax.set_title(
            display_name(provider), fontsize=11, color=TEXT_COLOR, fontweight="bold"
        )
        ax.set_xlabel("R (conversations per persona)", fontsize=9, color=TEXT_COLOR)
        ax.set_ylabel(
            "Standard Deviation of VERA Score score", fontsize=9, color=TEXT_COLOR
        )
        ax.tick_params(colors=TEXT_COLOR)
        ax.set_xticks(r_values)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(len(providers), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01, color=TEXT_COLOR)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"📈 Bootstrap comparison plot saved to: {output_path}")


def compute_judge_user_breakdown(
    df: pd.DataFrame,
    persona_metadata: Dict[str, Dict[str, str]],
    p_target: int = 100,
    r_target: int = 4,
    u_values: Optional[List[str]] = None,
    j_values: Optional[List[str]] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    For every (user_llm, judge_model, provider_llm) combination compute a single
    VERA score at the fixed best-effort (P, R) point.

    This answers: "Does my choice of judge or user LLM systematically shift the
    score for a given provider?"

    Returns a DataFrame with columns:
        user_llm, judge_model, provider_llm, P, R, n_samples, mean_score, std_score
    """
    # Limit to max_turns=20
    if "max_turns" in df.columns:
        df_raw = df[df["max_turns"] == 20].copy()
        assert isinstance(df_raw, pd.DataFrame)
        df = df_raw

    groupby_cols = [
        c for c in ["user_llm", "judge_model", "provider_llm"] if c in df.columns
    ]
    results: List[Dict[str, Any]] = []

    for group_key, group_df_raw in df.groupby(groupby_cols, dropna=False):
        if isinstance(group_df_raw, pd.Series):
            continue
        group_df: pd.DataFrame = group_df_raw

        # Unpack key
        key = list(group_key) if isinstance(group_key, tuple) else [group_key]
        col_map = {
            col: (key[i] if i < len(key) else "") for i, col in enumerate(groupby_cols)
        }
        user_llm = str(col_map.get("user_llm", ""))
        judge_model = str(col_map.get("judge_model", ""))
        provider_llm = str(col_map.get("provider_llm", ""))

        if u_values and user_llm not in u_values:
            continue
        if j_values and judge_model not in j_values:
            continue
        if "persona_name" not in group_df.columns:
            continue

        personas = list(group_df["persona_name"].unique())
        p_use = min(p_target, len(personas))

        sampled_personas = stratify_personas_by_risk_disclosure(
            personas,
            persona_metadata,
            p_use,
            random_seed=random_seed,
        )

        persona_subset_raw = group_df[group_df["persona_name"].isin(sampled_personas)]
        assert isinstance(persona_subset_raw, pd.DataFrame)
        persona_subset: pd.DataFrame = persona_subset_raw

        conversations_by_persona: Dict[str, Dict[str, pd.DataFrame]] = {}
        for persona in sampled_personas:
            pconv_raw = persona_subset[persona_subset["persona_name"] == persona].copy()
            if isinstance(pconv_raw, pd.DataFrame) and not pconv_raw.empty:
                convs = group_rows_by_conversation(pconv_raw)
                if convs:
                    conversations_by_persona[str(persona)] = convs

        if not conversations_by_persona:
            continue

        min_convs = min(len(c) for c in conversations_by_persona.values())
        r_use = min(r_target, min_convs)

        # Use all judge rows per conversation so both judge models are pooled.
        min_rows_per_conv = min(
            len(conv_df)
            for persona_convs in conversations_by_persona.values()
            for conv_df in persona_convs.values()
        )

        try:
            sample_dfs = partition_conversations(
                conversations_by_persona,
                r_conversations=r_use,
                j_iterations=min_rows_per_conv,
                random_seed=random_seed,
            )
        except ValueError as e:
            print(f"   SKIPPED {user_llm}/{judge_model}/{provider_llm}: {e}")
            continue

        sample_scores: List[float] = []
        for sample_df in sample_dfs:
            sc = calculate_sample_scores(sample_df)
            sample_scores.append(sc["overall_score"])

        if not sample_scores:
            continue

        mean_s = sum(sample_scores) / len(sample_scores)
        std_s = (
            sum((s - mean_s) ** 2 for s in sample_scores) / len(sample_scores)
        ) ** 0.5

        results.append(
            {
                "user_llm": user_llm,
                "judge_model": judge_model,
                "provider_llm": provider_llm,
                "P": p_use,
                "R": r_use,
                "n_samples": len(sample_scores),
                "mean_score": round(mean_s, 2),
                "std_score": round(std_s, 2),
            }
        )
        print(
            f"   {user_llm} / {judge_model} / {provider_llm}: "
            f"mean={mean_s:.1f} std={std_s:.2f} (n={len(sample_scores)})"
        )

    return pd.DataFrame(results)


def plot_judge_user_breakdown(
    breakdown_df: pd.DataFrame,
    output_path: Path,
    title: str = "Score Sensitivity to User LLM and Judge LLM",
) -> None:
    """
    2×2 grid of grouped-bar plots, one subplot per provider.

    Within each subplot, each (user_llm × judge_model) combination is a bar.
    Bars for the same user LLM share the same colour family; the two judge
    models within that family use a lighter vs. darker shade.  Error bars = ±1 std.
    """
    if breakdown_df.empty:
        print("⚠️  No breakdown data to plot")
        return

    providers = sorted(breakdown_df["provider_llm"].unique())
    user_llms = sorted(breakdown_df["user_llm"].unique())
    judge_models = sorted(breakdown_df["judge_model"].unique())

    # Colour families: 2 shades per user LLM
    user_families = [
        ("#2196F3", "#0D47A1"),  # blues  — first user LLM
        ("#FF9800", "#E65100"),  # oranges — second user LLM
        ("#4CAF50", "#1B5E20"),  # greens  — third (if any)
        ("#E91E63", "#880E4F"),  # pinks
    ]
    # map user_llm → (light_color, dark_color)
    user_colors: dict[str, tuple[str, str]] = {
        u: user_families[i % len(user_families)] for i, u in enumerate(user_llms)
    }
    # map judge_model index → which shade (0=light, 1=dark)
    judge_shade_idx: dict[str, int] = {j: i for i, j in enumerate(judge_models)}

    # Build ordered combos and assign bar colours
    combos: list[tuple[str, str]] = [(u, j) for u in user_llms for j in judge_models]
    combo_colors = [user_colors[u][judge_shade_idx[j] % 2] for u, j in combos]

    # Short display labels
    def short(name: str) -> str:
        return display_name(name)

    combo_labels = [f"{short(u)}\n{short(j)}" for u, j in combos]
    x_pos = np.arange(len(combos))
    bar_width = 0.55

    n_prov = len(providers)
    ncols = 2
    nrows = (n_prov + 1) // 2

    # Shared y-limits — start at 0 so bar heights are directly comparable
    y_lo = 0.0
    y_hi = float((breakdown_df["mean_score"] + breakdown_df["std_score"]).max()) + 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), squeeze=False)
    fig.patch.set_facecolor(BG_COLOR)

    from matplotlib.patches import Patch

    for idx, provider in enumerate(providers):
        row_idx, col_idx = divmod(idx, ncols)
        ax = axes[row_idx][col_idx]

        prov_df_raw = breakdown_df[breakdown_df["provider_llm"] == provider]
        assert isinstance(prov_df_raw, pd.DataFrame)
        prov_df: pd.DataFrame = prov_df_raw

        for ci, (user, judge) in enumerate(combos):
            mask = (prov_df["user_llm"] == user) & (prov_df["judge_model"] == judge)
            row_raw = prov_df[mask]
            assert isinstance(row_raw, pd.DataFrame)
            row = row_raw

            if row.empty:
                continue

            mean_v = float(row["mean_score"].iloc[0])
            std_v = float(row["std_score"].iloc[0])
            color = combo_colors[ci]

            ax.bar(
                x_pos[ci],
                mean_v,
                width=bar_width,
                color=color,
                alpha=0.85,
                yerr=std_v,
                capsize=5,
                error_kw={"elinewidth": 1.5, "ecolor": TEXT_COLOR},
                zorder=3,
            )
            ax.text(
                x_pos[ci],
                mean_v + std_v + 0.25,
                f"{mean_v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=TEXT_COLOR,
            )

        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(combo_labels, fontsize=8)
        ax.set_ylabel("VERA Overall Score", fontsize=10, color=TEXT_COLOR)
        ax.set_title(
            display_name(provider), fontsize=11, fontweight="bold", color=TEXT_COLOR
        )
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_facecolor("white")
        ax.tick_params(labelsize=8)

    # Shared legend (user LLM families) — only on last subplot
    legend_handles = []
    for u in user_llms:
        light, dark = user_colors[u]
        for j_idx, j in enumerate(judge_models):
            shade = dark if j_idx else light
            legend_handles.append(
                Patch(facecolor=shade, alpha=0.85, label=f"{short(u)} / {short(j)}")
            )
    # Place legend on the gemini subplot, upper left
    gemini_idx = next(
        (i for i, p in enumerate(providers) if "gemini" in p.lower()),
        0,  # fallback to first provider
    )
    g_row, g_col = divmod(gemini_idx, ncols)
    axes[g_row][g_col].legend(
        handles=legend_handles,
        fontsize=8,
        loc="upper left",
        title="user / judge",
    )

    # Hide unused subplots
    for empty_idx in range(n_prov, nrows * ncols):
        r, c = divmod(empty_idx, ncols)
        axes[r][c].set_visible(False)

    p_vals = breakdown_df["P"].unique()
    r_vals = breakdown_df["R"].unique()
    p_str = "/".join(str(v) for v in sorted(p_vals))
    r_str = "/".join(str(v) for v in sorted(r_vals))
    fig.suptitle(
        f"{title}  (P={p_str}, R={r_str})",
        fontsize=13,
        fontweight="bold",
        color=TEXT_COLOR,
        y=1.01,
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"📈 Judge/user breakdown plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze VERA score variability across sampling strategies"
    )

    parser.add_argument(
        "--input-csv",
        "-i",
        default="concatenated_heor_paper2_results.csv",
        help="Input concatenated results CSV file (default: concatenated_heor_paper2_results.csv)",
    )

    parser.add_argument(
        "--personas-tsv",
        default="data/personas.tsv",
        help="Path to personas.tsv file (default: data/personas.tsv)",
    )

    parser.add_argument(
        "--p-values",
        "-p",
        nargs="+",
        type=int,
        default=[10, 25, 50, 100],
        help="P values: number of personas (default: 10 25 50 100)",
    )

    parser.add_argument(
        "--r-values",
        "-r",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7, 8],
        help="R values: conversations per persona (default: 1 2 3 4 5 6 7 8)",
    )

    parser.add_argument(
        "--t-values",
        "-t",
        nargs="+",
        type=int,
        default=[20],
        help="T values: max_turns per conversation (default: 20 — only t=20 analysed)",
    )

    parser.add_argument(
        "--u-values",
        "-u",
        nargs="+",
        type=str,
        default=["gpt 5 2", "opus 4 5"],
        help="U values: persona/user models (default: gpt 5 2 opus 4 5)",
    )

    parser.add_argument(
        "--j-values",
        "-j",
        nargs="+",
        type=str,
        default=["gpt-4o", "claude-sonnet-4-5-20250929"],
        help="J values: judge models (default: gpt-4o claude-sonnet-4-5-20250929)",
    )

    parser.add_argument(
        "--random-seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--n-bootstrap",
        "-b",
        type=int,
        default=200,
        help="Bootstrap iterations for std estimation (default: 200)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="score_variability/score_variability_analysis.csv",
        help="Output CSV file path",
    )

    parser.add_argument(
        "--plot",
        default=None,
        help="Output plot file path (default: same as output with .png extension)",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the plot",
    )

    parser.add_argument(
        "--plots-only",
        action="store_true",
        help=(
            "Skip the bootstrap analysis entirely; load the existing output CSV "
            "and regenerate all plots. Useful for iterating on visualisations "
            "without re-running the slow bootstrap."
        ),
    )

    args = parser.parse_args()

    input_csv_path = Path(args.input_csv)
    if not input_csv_path.exists():
        print(f"❌ Error: Input CSV file not found: {input_csv_path}")
        return 1

    personas_tsv_path = Path(args.personas_tsv)
    if not personas_tsv_path.exists():
        print(f"❌ Error: Personas TSV not found: {personas_tsv_path}")
        return 1

    output_path = Path(args.output)

    if args.plots_only:
        # ── Fast path: load existing CSV, skip bootstrap ─────────────────────
        if not output_path.exists():
            print(
                f"❌ --plots-only requested but output CSV not found: {output_path}\n"
                f"   Run without --plots-only first to generate it."
            )
            return 1
        print(f"📂 Loading cached results from: {output_path}")
        results_df = pd.read_csv(output_path)
        print(f"✅ Loaded {len(results_df)} rows")

        # We still need the raw df and persona_metadata for the judge/user
        # breakdown computation (which has its own internal sampling).
        print(f"\n📂 Loading concatenated results from: {input_csv_path}")
        df = load_concatenated_results(input_csv_path)
        print(f"\n📋 Loading persona metadata from: {personas_tsv_path}")
        persona_metadata = load_persona_metadata(personas_tsv_path)
        print(f"✅ Loaded metadata for {len(persona_metadata)} personas")
    else:
        # ── Full path: run bootstrap, save CSV ───────────────────────────────
        print(f"📂 Loading concatenated results from: {input_csv_path}")
        df = load_concatenated_results(input_csv_path)

        print(f"\n📋 Loading persona metadata from: {personas_tsv_path}")
        persona_metadata = load_persona_metadata(personas_tsv_path)
        print(f"✅ Loaded metadata for {len(persona_metadata)} personas")

        print("\n🔬 Running sampling analysis...")
        print(f"   P values: {args.p_values}")
        print(f"   R values: {args.r_values}")
        print(f"   T values: {args.t_values}")
        print(f"   U values: {args.u_values}")
        print(f"   J values: {args.j_values}")
        print(f"   Random seed: {args.random_seed}")
        print(f"   Bootstrap iterations: {args.n_bootstrap}")

        results_df = run_sampling_analysis(
            df,
            persona_metadata,
            p_values=args.p_values,
            r_values=args.r_values,
            t_values=args.t_values,
            u_values=args.u_values,
            j_values=args.j_values,
            random_seed=args.random_seed,
            n_bootstrap=args.n_bootstrap,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")

    # Generate plots
    if not args.no_plot:
        plot_path = Path(args.plot) if args.plot else output_path.with_suffix(".png")

        convergence_path = plot_path.with_stem(plot_path.stem + "_score_convergence")
        plot_score_convergence_by_provider(
            results_df, convergence_path, bootstrap_type="persona"
        )

        threshold_path = plot_path.with_stem(plot_path.stem + "_std_threshold")
        plot_std_threshold(results_df, threshold_path, bootstrap_type="persona")

        threshold_combined_path = plot_path.with_stem(
            plot_path.stem + "_std_threshold_combined"
        )
        plot_std_threshold_combined(
            results_df, threshold_combined_path, bootstrap_type="persona"
        )

        # CI-width version (pooled)
        ci_combined_path = plot_path.with_stem(
            plot_path.stem + "_ci_width_threshold_combined"
        )
        plot_std_threshold_combined(
            results_df, ci_combined_path, bootstrap_type="persona", use_ci_width=True
        )

        for judge_substr, judge_suffix in [
            ("gpt-4o", "gpt4o"),
            ("claude-sonnet-4-5-20250929", "sonnet"),
        ]:
            jpath = plot_path.with_stem(
                plot_path.stem + f"_std_threshold_combined_{judge_suffix}"
            )
            plot_std_threshold_combined(
                results_df,
                jpath,
                bootstrap_type="persona",
                judge_filter=judge_substr,
            )

            # CI-width version (per judge)
            ci_jpath = plot_path.with_stem(
                plot_path.stem + f"_ci_width_threshold_combined_{judge_suffix}"
            )
            plot_std_threshold_combined(
                results_df,
                ci_jpath,
                bootstrap_type="persona",
                judge_filter=judge_substr,
                use_ci_width=True,
            )

        bootstrap_cmp_path = plot_path.with_stem(
            plot_path.stem + "_bootstrap_type_comparison"
        )
        plot_bootstrap_type_comparison(results_df, bootstrap_cmp_path)

        # Cost-efficiency plots: all judges + one per judge
        _cost_eff_judges: list[tuple[str | None, str]] = [(None, "")]
        if "judge_model" in df.columns:
            for _jm in sorted(df["judge_model"].dropna().unique()):
                _jm_str = str(_jm)
                _jm_suffix = "_" + display_name(_jm_str).lower().replace(
                    " ", "_"
                ).replace("-", "").replace(".", "")
                _cost_eff_judges.append((_jm_str, _jm_suffix))

        for _jf, _jsuffix in _cost_eff_judges:
            _jlabel = "All Judges" if _jf is None else display_name(_jf)
            print(f"\n📈 Cost-efficiency plots ({_jlabel})...")

            cost_eff_path = plot_path.with_stem(
                plot_path.stem + f"_cost_efficiency{_jsuffix}"
            )
            plot_cost_efficiency(
                results_df,
                cost_eff_path,
                title=f"Evaluation Cost Efficiency: Adding Personas vs. Conversations ({_jlabel})",
                judge_filter=_jf,
            )

            cost_eff_ci_path = plot_path.with_stem(
                plot_path.stem + f"_cost_efficiency_ci_width{_jsuffix}"
            )
            plot_cost_efficiency(
                results_df,
                cost_eff_ci_path,
                title=f"Evaluation Cost Efficiency (95% CI Width): Adding Personas vs. Conversations ({_jlabel})",
                judge_filter=_jf,
                use_ci_width=True,
            )

            cost_eff_std_path = plot_path.with_stem(
                plot_path.stem + f"_cost_efficiency_std{_jsuffix}"
            )
            plot_cost_efficiency(
                results_df,
                cost_eff_std_path,
                # title=f"Evaluation Cost Efficiency (Std): Adding Personas vs. Conversations ({_jlabel})",
                title="VERA-MH Evaluation Parameters: Adding Profiles vs. Conversations",
                judge_filter=_jf,
                use_std=True,
            )

            cost_eff_by_p_path = plot_path.with_stem(
                plot_path.stem + f"_cost_efficiency_by_p{_jsuffix}"
            )
            plot_cost_efficiency_by_p(
                results_df,
                cost_eff_by_p_path,
                title=f"Score Range vs. Total Conversations by Provider Model: Fixed P, Varying R ({_jlabel})",
                judge_filter=_jf,
            )

            cost_eff_by_p_std_path = plot_path.with_stem(
                plot_path.stem + f"_cost_efficiency_by_p_std{_jsuffix}"
            )
            plot_cost_efficiency_by_p(
                results_df,
                cost_eff_by_p_std_path,
                title=f"Score Std vs. Total Conversations by Provider Model: Fixed P, Varying R ({_jlabel})",
                judge_filter=_jf,
                use_std=True,
            )

            cost_eff_by_p_ci_path = plot_path.with_stem(
                plot_path.stem + f"_cost_efficiency_by_p_ci_width{_jsuffix}"
            )
            plot_cost_efficiency_by_p(
                results_df,
                cost_eff_by_p_ci_path,
                title=f"95% CI Width vs. Total Conversations by Provider Model: Fixed P, Varying R ({_jlabel})",
                judge_filter=_jf,
                use_ci_width=True,
            )

            cost_eff_by_p_comb_path = plot_path.with_stem(
                plot_path.stem + f"_cost_efficiency_by_p_combined{_jsuffix}"
            )
            plot_cost_efficiency_by_p_combined(
                results_df,
                cost_eff_by_p_comb_path,
                title=f"Score Range vs. Total Conversations: Fixed P, Varying R ({_jlabel})",
                judge_filter=_jf,
            )

            cost_eff_by_r_path = plot_path.with_stem(
                plot_path.stem + f"_cost_efficiency_by_r{_jsuffix}"
            )
            plot_cost_efficiency_by_r(
                results_df,
                cost_eff_by_r_path,
                title=f"Score Range vs. Total Conversations by Provider Model: Fixed R, Varying P ({_jlabel})",
                judge_filter=_jf,
            )

        print("\n🔬 Computing judge × user LLM breakdown (P=100, R=4)...")
        breakdown_df = compute_judge_user_breakdown(
            df,
            persona_metadata,
            p_target=100,
            r_target=4,
            u_values=args.u_values,
            j_values=args.j_values,
            random_seed=args.random_seed,
        )
        breakdown_csv = output_path.with_stem(
            output_path.stem + "_judge_user_breakdown"
        )
        breakdown_df.to_csv(breakdown_csv, index=False)
        print(f"✅ Breakdown saved to: {breakdown_csv}")

        breakdown_plot = plot_path.with_stem(plot_path.stem + "_judge_user_breakdown")
        plot_judge_user_breakdown(breakdown_df, breakdown_plot)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
