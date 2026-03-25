#!/usr/bin/env python3
"""
Check all conversations in conversations/HEOR_FIXED/ for role confusion using
the LangSmith prompt "vera-role-confusion-check".

Results are counted by persona LLM (p_), provider agent LLM (a_), and max turns (t).

Usage:
    python scripts/role_confusion_check.py
    python scripts/role_confusion_check.py --conversations-dir conversations/HEOR_FIXED
    python scripts/role_confusion_check.py -o results/role_confusion.csv
"""

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Disable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from langsmith import Client  # noqa: F401

    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

PROMPT_NAME = "vera-role-confusion-check"

# Folder name pattern: p_{persona}__a_{agent}__t{turns}__r{run}__{datetime}
FOLDER_PATTERN = re.compile(
    r"^p_(?P<persona>.+?)__a_(?P<agent>.+?)__t(?P<turns>\d+)__r(?P<run>\d+)__(?P<datetime>.+)$"
)


class RoleConfusionOutput(BaseModel):
    role_confusion: bool


def parse_folder_name(folder_name: str) -> dict:
    """Parse persona LLM, agent LLM, max_turns, run, and datetime from a folder name."""
    match = FOLDER_PATTERN.match(folder_name)
    if match:
        return {
            "persona_llm": match.group("persona"),
            "agent_llm": match.group("agent"),
            "max_turns": int(match.group("turns")),
            "run": int(match.group("run")),
            "datetime": match.group("datetime"),
            "run_id": folder_name,
        }
    return {
        "persona_llm": "unknown",
        "agent_llm": "unknown",
        "max_turns": 0,
        "run": 0,
        "datetime": "unknown",
        "run_id": folder_name,
    }


def build_langsmith_client() -> Optional["Client"]:
    """Build a LangSmith Client using environment variables."""
    if not LANGSMITH_AVAILABLE:
        return None
    endpoint = (
        os.environ.get("LANGSMITH_ENDPOINT")
        or os.environ.get("LANGCHAIN_ENDPOINT")
        or os.environ.get("LANGCHAIN_API_URL")
    )
    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    client_kwargs = {}
    if endpoint:
        client_kwargs["api_url"] = endpoint
    if api_key:
        client_kwargs["api_key"] = api_key
    return Client(**client_kwargs) if client_kwargs else Client()  # type: ignore[return-value]


def _replace_azure_endpoint(obj: object, endpoint: str) -> int:
    """Recursively replace all azure_endpoint values in a manifest dict. Returns count replaced."""
    count = 0
    if isinstance(obj, dict):
        if "azure_endpoint" in obj:
            obj["azure_endpoint"] = endpoint
            count += 1
        for v in obj.values():
            count += _replace_azure_endpoint(v, endpoint)
    elif isinstance(obj, list):
        for item in obj:
            count += _replace_azure_endpoint(item, endpoint)
    return count


def load_prompt_from_langsmith(client: "Client", prompt_name: str) -> Optional[object]:
    """
    Load a prompt from LangSmith, handling AzureChatOpenAI deserialization and
    applying the structured-output transformation that pull_prompt() normally does.

    Returns a ready-to-invoke Runnable, or None on failure.
    """
    import json as _json

    from langchain_core.load import loads
    from langchain_core.runnables import Runnable

    # Debug: show credentials being used
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_endpoint_override = os.environ.get("AZURE_OPENAI_API_URL") or os.environ.get(
        "AZURE_OPENAI_ENDPOINT"
    )
    print(
        f"   🔍 AZURE_OPENAI_API_KEY: {'***' + azure_key[-4:] if azure_key and len(azure_key) > 4 else 'NOT SET'}"
    )
    print(
        f"   🔍 Azure endpoint override: {azure_endpoint_override or 'none (will use baked value)'}"
    )

    try:
        prompt_commit = client.pull_prompt_commit(prompt_name, include_model=True)

        # Deep-copy the manifest and recursively override all azure_endpoint fields
        manifest = _json.loads(_json.dumps(prompt_commit.manifest))
        if azure_endpoint_override:
            replaced = _replace_azure_endpoint(manifest, azure_endpoint_override)
            print(f"   🔍 Replaced {replaced} azure_endpoint field(s) in manifest")

        # Deserialize with allowed_objects='all' to permit AzureChatOpenAI
        prompt = loads(_json.dumps(manifest), allowed_objects="all")

        # Apply the post-deserialization structured-output transformation that
        # pull_prompt() normally performs (StructuredPrompt | model re-pipe).
        from langchain_core.language_models import BaseLanguageModel
        from langchain_core.prompts.structured import StructuredPrompt
        from langchain_core.runnables.base import RunnableBinding, RunnableSequence

        if (
            isinstance(prompt, RunnableSequence)
            and isinstance(prompt.first, StructuredPrompt)
            and len(list(prompt.steps)) == 2
        ):
            last = prompt.last
            if isinstance(last, RunnableBinding) and isinstance(
                last.bound, BaseLanguageModel
            ):
                seq = prompt.first | last.bound  # type: ignore[operator]
                seq_steps = list(seq.steps) if isinstance(seq, RunnableSequence) else []  # type: ignore[union-attr]
                if len(seq_steps) == 3:
                    rebound_llm = seq_steps[1]
                    prompt = RunnableSequence(
                        prompt.first,
                        rebound_llm.bind(**{**last.kwargs}),  # type: ignore[union-attr]
                        seq_steps[2],
                    )
                else:
                    prompt = seq
            elif isinstance(last, BaseLanguageModel):
                prompt = prompt.first | last  # type: ignore[operator]

        if not isinstance(prompt, Runnable):
            print(f"⚠️  Prompt is not a Runnable (type: {type(prompt)})")
            return None

        return prompt

    except Exception as e:
        print(f"❌ Failed to load prompt '{prompt_name}': {e}")
        import traceback

        traceback.print_exc()
        return None


async def evaluate_conversation(
    prompt: object,
    conversation_text: str,
    conversation_id: str,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> Optional[RoleConfusionOutput]:
    """Invoke the role-confusion prompt on one conversation and return structured output.

    Retries up to max_retries times on any error or unexpected output type.
    """
    import asyncio

    from langchain_core.runnables import Runnable

    if not isinstance(prompt, Runnable):
        return None

    for attempt in range(1, max_retries + 1):
        try:
            result_raw = await prompt.ainvoke({"conversation": conversation_text})
            if isinstance(result_raw, RoleConfusionOutput):
                return result_raw
            elif isinstance(result_raw, dict):
                return RoleConfusionOutput(**result_raw)
            else:
                raise ValueError(
                    f"Unexpected result type: {type(result_raw)} — {result_raw}"
                )
        except Exception as e:
            if attempt < max_retries:
                print(f"   ⚠️  Attempt {attempt}/{max_retries} failed: {e}")
                print(f"   ⏳ Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            else:
                print(
                    f"   ❌ All {max_retries} attempts failed for {conversation_id}: {e}"
                )

    return None


async def process_conversations(
    conversations_base_dir: Path,
) -> pd.DataFrame:
    """Walk all run folders, evaluate every .txt file, and return a results DataFrame."""
    if not conversations_base_dir.exists():
        print(
            f"❌ Conversations directory not found: {conversations_base_dir.absolute()}"
        )
        return pd.DataFrame()

    if not LANGSMITH_AVAILABLE:
        print("❌ LangSmith is not installed. Install with: uv add langsmith")
        return pd.DataFrame()

    client = build_langsmith_client()
    if client is None:
        print("❌ Could not build LangSmith client.")
        return pd.DataFrame()

    print(f"🔌 Loading prompt '{PROMPT_NAME}' from LangSmith...")
    prompt = load_prompt_from_langsmith(client, PROMPT_NAME)
    if prompt is None:
        print("❌ Could not load prompt. Aborting.")
        return pd.DataFrame()
    print(f"✅ Prompt loaded: {type(prompt)}")

    # Collect all (run_folder, txt_file) pairs
    run_dirs = [d for d in sorted(conversations_base_dir.iterdir()) if d.is_dir()]
    print(f"\n📁 Found {len(run_dirs)} run folders in {conversations_base_dir}")

    results = []
    total_files = sum(len(list(d.glob("*.txt"))) for d in run_dirs)
    processed = 0
    attempted = 0
    errors = 0

    debug_limit = 0  # Set to a positive integer to limit LLM calls for debugging
    print(f"📄 Total conversation files: {total_files}\n")

    for run_dir in run_dirs:
        meta = parse_folder_name(run_dir.name)
        txt_files = sorted(run_dir.glob("*.txt"))
        if not txt_files:
            continue

        for txt_file in txt_files:
            conversation_id = f"{run_dir.name}/{txt_file.name}"
            print(f"  ▶ {conversation_id}", end=" ... ", flush=True)

            try:
                conversation_text = txt_file.read_text(encoding="utf-8")
            except Exception as e:
                print(f"❌ read error: {e}")
                errors += 1
                continue

            if debug_limit and attempted >= debug_limit:
                print(
                    f"  ⏭  Debug limit reached ({debug_limit}), skipping remaining files"
                )
                break

            attempted += 1
            result = await evaluate_conversation(
                prompt, conversation_text, conversation_id
            )

            if result is not None:
                row = {
                    **meta,
                    "filename": txt_file.name,
                    "conversation_path": str(txt_file),
                    "role_confusion": result.role_confusion,
                }
                results.append(row)
                processed += 1
                print(f"role_confusion={result.role_confusion}")
            else:
                errors += 1
                print("❌ evaluation failed")

        if debug_limit and attempted >= debug_limit:
            break

    print(
        f"\n✅ Processed: {processed}  ❌ Errors: {errors}  📋 Attempted: {attempted}"
    )
    return pd.DataFrame(results) if results else pd.DataFrame()


def print_summary(df: pd.DataFrame) -> None:
    """Print counts of role_confusion=True broken down by persona_llm, agent_llm, and max_turns."""
    if df.empty:
        print("No results to summarise.")
        return

    total = len(df)
    flagged = int(df["role_confusion"].sum())

    print("\n" + "=" * 80)
    print("ROLE CONFUSION SUMMARY")
    print("=" * 80)
    print(f"\nTotal conversations evaluated : {total}")
    print(f"Role confusion flagged (True) : {flagged}  ({100 * flagged / total:.1f}%)")

    # --- By persona LLM ---
    print("\n📊 Flagged by Persona LLM (p_):")
    by_persona = (
        df.groupby("persona_llm")["role_confusion"]
        .agg(flagged="sum", total="count")
        .assign(pct=lambda x: (100 * x["flagged"] / x["total"]).round(1))
        .sort_values("flagged", ascending=False)
    )
    print(by_persona.to_string())

    # --- By agent LLM ---
    print("\n📊 Flagged by Provider Agent LLM (a_):")
    by_agent = (
        df.groupby("agent_llm")["role_confusion"]
        .agg(flagged="sum", total="count")
        .assign(pct=lambda x: (100 * x["flagged"] / x["total"]).round(1))
        .sort_values("flagged", ascending=False)
    )
    print(by_agent.to_string())

    # --- By max_turns ---
    print("\n📊 Flagged by Max Turns (t):")
    by_turns = (
        df.groupby("max_turns")["role_confusion"]
        .agg(flagged="sum", total="count")
        .assign(pct=lambda x: (100 * x["flagged"] / x["total"]).round(1))
        .sort_values("max_turns")
    )
    print(by_turns.to_string())

    # --- Crosstab: persona × agent ---
    print("\n📋 Flagged count: Persona LLM × Agent LLM")
    try:
        ct = (
            pd.crosstab(
                df["persona_llm"],
                df["agent_llm"],
                values=df["role_confusion"].astype(int),
                aggfunc="sum",
            )
            .fillna(0)
            .astype(int)
        )
        print(ct.to_string())
    except Exception as e:
        print(f"   ⚠️  Could not generate crosstab: {e}")

    # # --- Flagged conversations ---
    # flagged_df = df[df["role_confusion"]]
    # if len(flagged_df) > 0:
    #     print(f"\n🚨 Flagged conversations ({len(flagged_df)}):")
    #     for _, row in flagged_df.iterrows():
    #         print(f"\n   📄 {row['filename']}")
    #         print(f"      Persona LLM : {row['persona_llm']}")
    #         print(f"      Agent LLM   : {row['agent_llm']}")
    #         print(f"      Max turns   : {row['max_turns']}")
    #         print(f"      Path        : {row['conversation_path']}")
    # else:
    #     print("\n✅ No conversations flagged for role confusion.")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check conversations for role confusion using LangSmith"
    )
    parser.add_argument(
        "--conversations-dir",
        type=Path,
        default=Path("conversations/HEOR_FIXED"),
        help="Directory containing run folders with .txt conversation files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to save results CSV (default: role_confusion_results.csv)",
    )
    args = parser.parse_args()

    df = await process_conversations(args.conversations_dir)

    if df.empty:
        print("No results.")
        return

    print_summary(df)

    output_path = args.output or Path("role_confusion_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to {output_path.absolute()}")


if __name__ == "__main__":
    asyncio.run(main())
