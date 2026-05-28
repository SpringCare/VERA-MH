#!/usr/bin/env python3

import asyncio
import logging
import os
import time
import uuid
from asyncio import Queue
from datetime import datetime
from typing import AbstractSet, Any, Dict, List, Optional, Tuple

from llm_clients import LLMFactory
from llm_clients.llm_interface import LLMGenerationFailed, Role
from utils.conversation_layout import resolve_conversation_input
from utils.logging_utils import (
    cleanup_logger,
    log_conversation_end,
    log_conversation_start,
    log_conversation_turn,
    setup_conversation_logger,
)
from utils.naming import (
    TRANSCRIPT_RUN_SUFFIX_RE,
    persona_token_for_transcript_stem,
)

from .conversation_simulator import ConversationSimulator
from .utils import load_prompts_from_csv


class ConversationRunner:
    """Handles running LLM conversations with logging and file management."""

    def __init__(
        self,
        persona_model_config: Dict[str, Any],
        agent_model_config: Dict[str, Any],
        run_id: str,
        max_turns: int = 6,
        runs_per_prompt: int = 3,
        folder_name: str = "conversations",
        max_concurrent: Optional[int] = None,
        max_total_words: Optional[int] = None,
        max_personas: Optional[int] = None,
        persona_speaks_first: bool = True,
        session_types: Optional[List[str]] = None,
        resume: bool = False,
    ):
        self.persona_model_config = persona_model_config
        self.agent_model_config = agent_model_config
        self.max_turns = max_turns
        self.runs_per_prompt = runs_per_prompt
        self.folder_name = folder_name
        self.run_id = run_id

        # Limit concurrent conversations to avoid overwhelming the server
        # Default: None - run all conversations concurrently
        self.max_concurrent = max_concurrent
        self.max_total_words = max_total_words
        self.max_personas = max_personas
        self.persona_speaks_first = persona_speaks_first
        self.session_types = session_types
        self.resume = resume

        # folder_name: p_run root (or legacy flat). .txt and logs go in
        # transcripts_dir (e.g. p_run/conversations/), not folder_name.
        transcripts_dir, _, _ = resolve_conversation_input(folder_name)
        self.transcripts_dir = transcripts_dir
        self.logs_dir = os.path.join(transcripts_dir, "logs")

    @staticmethod
    def _resolve_persona_safe_from_stem(
        stem: str, persona_safe_names: AbstractSet[str]
    ) -> Optional[str]:
        """
        Pick the longest persona_safe in persona_safe_names that matches this stem.

        Transcript stem (after tag_) is {persona_safe}_{model_name}; model_name may
        contain underscores, so we match known persona_safe names instead of splitting
        on underscores.

        Example: stem ``Anna_gemini-2.5-flash`` with ``{"Ann", "Anna"}`` in the set
        matches both prefixes; returns ``Anna`` (longest).
        """
        candidates = [
            p for p in persona_safe_names if stem == p or stem.startswith(f"{p}_")
        ]
        if not candidates:
            return None
        return max(candidates, key=len)

    def _parse_transcript_filename_for_resume(
        self, filename: str, persona_safe_names: AbstractSet[str]
    ) -> Optional[tuple[str, int, Optional[int], Optional[str]]]:
        """
        Parse transcript basename using known persona_safe names (tag is discarded).

        Supports:
        - `{tag}_{persona_safe}_{model}_run{N}.txt`
        - `{tag}_{persona_safe}_{model}_run{N}_s{session_index}_{session_type}.txt`
        """
        if not filename.endswith(".txt"):
            return None
        parts = filename.split("_", 1)
        if len(parts) != 2:
            return None
        suffix = parts[1]
        match = TRANSCRIPT_RUN_SUFFIX_RE.search(suffix)
        if not match:
            return None
        run = int(match.group("run"))
        session_index = (
            int(match.group("session_index")) if match.group("session_index") else None
        )
        session_type = match.group("session_type")
        stem = suffix[: match.start()]
        persona_safe = self._resolve_persona_safe_from_stem(stem, persona_safe_names)
        if persona_safe is None:
            return None
        return (persona_safe, run, session_index, session_type)

    def _list_existing_conversations(
        self, persona_safe_names: AbstractSet[str]
    ) -> list[tuple[str, int, Optional[int], Optional[str]]]:
        """
        List parsed transcript entries for resume.

        Each entry is (persona_safe, run_number, session_index, session_type).
        Single-session files use None for the session fields.
        """
        out: list[tuple[str, int, Optional[int], Optional[str]]] = []
        if not os.path.isdir(self.transcripts_dir):
            return out

        for filename in os.listdir(self.transcripts_dir):
            parsed = self._parse_transcript_filename_for_resume(
                filename, persona_safe_names
            )
            if parsed is None:
                continue
            out.append(parsed)
        return out

    def _build_resume_state(
        self, existing: list[tuple[str, int, Optional[int], Optional[str]]]
    ) -> tuple[set[tuple[str, int]], dict[tuple[str, int], set[tuple[int, str]]]]:
        """Derive single-session keys and per-run completed session sets."""
        complete_runs: set[tuple[str, int]] = set()
        sessions_by_run: dict[tuple[str, int], set[tuple[int, str]]] = {}

        for persona_safe, run_number, session_index, session_type in existing:
            key = (persona_safe, run_number)
            if session_index is None or session_type is None:
                complete_runs.add(key)
                continue
            sessions_by_run.setdefault(key, set()).add((session_index, session_type))

        return complete_runs, sessions_by_run

    def _has_existing_transcript(
        self,
        persona_safe: str,
        run_number: int,
        complete_runs: set[tuple[str, int]],
        sessions_by_run: dict[tuple[str, int], set[tuple[int, str]]],
    ) -> bool:
        """Return True when this persona/run is fully complete for resume."""
        key = (persona_safe, run_number)
        if key in complete_runs:
            return True

        session_types = self.session_types
        if session_types and len(session_types) > 1:
            required = {(i, st) for i, st in enumerate(session_types, 1)}
            found = sessions_by_run.get(key, set())
            return required.issubset(found)

        return key in sessions_by_run

    def _session_transcript_exists(
        self,
        persona_safe: str,
        run_number: int,
        session_index: int,
        session_type: str,
        sessions_by_run: dict[tuple[str, int], set[tuple[int, str]]],
        complete_runs: set[tuple[str, int]],
    ) -> bool:
        """Return True when a specific multi-session transcript already exists."""
        key = (persona_safe, run_number)
        if key in complete_runs:
            return True
        return (session_index, session_type) in sessions_by_run.get(key, set())

    def _find_session_transcript_path(
        self,
        persona_safe: str,
        run_number: int,
        session_index: int,
        session_type: str,
        persona_safe_names: AbstractSet[str],
    ) -> Optional[str]:
        """Return the path to an existing session transcript, if any."""
        if not os.path.isdir(self.transcripts_dir):
            return None
        for filename in os.listdir(self.transcripts_dir):
            parsed = self._parse_transcript_filename_for_resume(
                filename, persona_safe_names
            )
            if parsed is None:
                continue
            p_safe, run, idx, stype = parsed
            if (
                p_safe == persona_safe
                and run == run_number
                and idx == session_index
                and stype == session_type
            ):
                return os.path.join(self.transcripts_dir, filename)
        return None

    @staticmethod
    def _job_result(
        *,
        index: int,
        llm1_model: str,
        llm1_prompt: str,
        run_number: int,
        sessions: List[Dict[str, Any]],
        duration: float,
        skipped: bool,
        error: Optional[str] = None,
        skip_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persona/run wrapper; each session is a separate entry in ``sessions``."""
        out: Dict[str, Any] = {
            "index": index,
            "llm1_model": llm1_model,
            "llm1_prompt": llm1_prompt,
            "run_number": run_number,
            "sessions": sessions,
            "duration": duration,
            "skipped": skipped,
        }
        if error is not None:
            out["error"] = error
        if skip_reason is not None:
            out["skip_reason"] = skip_reason
        return out

    def _create_conversation_jobs(
        self, persona_names: Optional[List[str]] = None
    ) -> List[Tuple[dict, int, int, int]]:
        """Create job tuples for all persona/run combinations."""
        personas = load_prompts_from_csv(persona_names, max_personas=self.max_personas)
        jobs: List[Tuple[dict, int, int, int]] = []
        conversation_index = 1
        for persona in personas:
            for run in range(1, self.runs_per_prompt + 1):
                jobs.append(
                    (
                        {
                            "prompt": persona["prompt"],
                            "name": persona["Name"],
                            "run": run,
                        },
                        self.max_turns,
                        conversation_index,
                        run,
                    )
                )
                conversation_index += 1
        return jobs

    async def _worker(
        self,
        worker_id: int,
        queue: Queue,
        results: List[Dict[str, Any]],
        total_jobs: int,
    ) -> None:
        """Worker that processes conversation generation jobs from a queue."""
        while True:
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            try:
                persona_config, max_turns, conversation_index, run_number = job
                conversation_name = persona_config.get("name", "Unknown")
                print(
                    f"[Worker {worker_id}] ({len(results) + 1}/{total_jobs}) "
                    f"{conversation_name} (run {run_number})"
                )
                result = await self.run_single_conversation(
                    persona_config=persona_config,
                    max_turns=max_turns,
                    conversation_index=conversation_index,
                    run_number=run_number,
                )
            except Exception as exc:
                # Don't let one failed job cancel all workers.
                # Keep result schema stable.
                result = self._job_result(
                    index=job[2] if len(job) > 2 else -1,
                    llm1_model=self.persona_model_config.get("model", "unknown"),
                    llm1_prompt=(
                        job[0].get("name", "Unknown")
                        if isinstance(job[0], dict)
                        else "Unknown"
                    ),
                    run_number=job[3] if len(job) > 3 else 0,
                    sessions=[],
                    duration=0.0,
                    skipped=True,
                    error=str(exc),
                )
                print(f"[Worker {worker_id}] Failed job: {result['error']}")
            finally:
                queue.task_done()

            results.append(result)

    async def run_single_conversation(
        self,
        persona_config: dict,
        max_turns: int,
        conversation_index: int,
        run_number: int,
        **kwargs: dict,
    ) -> Dict[str, Any]:
        """Run a simulated conversation (persona vs provider LLM),
          across one or more sessions.

        Without --sessions the session list has one element; the same loop handles
        both cases. The provider's ``prepare_sessions()`` may reorder or prepend
        session types (e.g. intake before coaching).

        Args:
            persona_config (dict): Must have "prompt" and "name". Persona LLM
                identity comes from ``self.persona_model_config`` (including ``model``).
            max_turns (int): Max conversation turns per session.
            conversation_index (int): Index in the batch of conversations.
            run_number (int): Run index for this prompt (e.g. 1 of runs_per_prompt).
            **kwargs: Unused; reserved for future use.

        Returns:
            Dict[str, Any]: index, llm1_model, llm1_prompt, run_number, duration,
            skipped, and ``sessions`` (one dict per session; length 1 without
            ``--sessions``). Per-session fields: conversation, filename, log_file,
            turns, early_termination, etc.
        """
        model_name = self.persona_model_config["model"]
        system_prompt = persona_config["prompt"]
        persona_name = persona_config["name"]

        tag = uuid.uuid4().hex[:6]

        workflow = self.agent_model_config.get("workflow", "")
        workflow_part = f"_{workflow}" if workflow else ""
        filename_base = (
            f"{tag}_{persona_name}_{model_name}{workflow_part}_run{run_number}"
        )
        os.makedirs(f"{self.folder_name}", exist_ok=True)
        os.makedirs(self.transcripts_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        logger: Optional[logging.Logger] = None
        persona: Optional[Any] = None
        agent: Optional[Any] = None
        start_time = time.time()
        result: Optional[Dict[str, Any]] = None

        try:
            persona = LLMFactory.create_llm(
                model_name=model_name,
                name=f"{model_name} {persona_name}",
                system_prompt=system_prompt,
                role=Role.PERSONA,
                **self.persona_model_config,
            )

            agent_kwargs = {
                k: v
                for k, v in self.agent_model_config.items()
                if k not in ("model", "name", "system_prompt")
            }

            agent = LLMFactory.create_llm(
                model_name=self.agent_model_config["model"],
                name=self.agent_model_config.get("name", "Provider"),
                system_prompt=self.agent_model_config.get(
                    "system_prompt", "You are a helpful AI assistant."
                ),
                role=Role.PROVIDER,
                **agent_kwargs,
            )

            await agent.setup()

            raw_sessions = self.session_types or [
                getattr(agent, "_session_type", "default")
            ]
            session_types_prepared = agent.prepare_sessions(raw_sessions)
            multi_session = len(session_types_prepared) > 1

            persona_safe = persona_token_for_transcript_stem(persona_name)
            complete_runs: set[tuple[str, int]] = set()
            sessions_by_run: dict[tuple[str, int], set[tuple[int, str]]] = {}
            if self.resume:
                existing = self._list_existing_conversations({persona_safe})
                complete_runs, sessions_by_run = self._build_resume_state(existing)

            simulator = ConversationSimulator(persona, agent)
            sessions: List[Dict[str, Any]] = []

            for i, session_type in enumerate(session_types_prepared, 1):
                if (
                    self.resume
                    and multi_session
                    and self._session_transcript_exists(
                        persona_safe,
                        run_number,
                        i,
                        session_type,
                        sessions_by_run,
                        complete_runs,
                    )
                ):
                    existing_path = self._find_session_transcript_path(
                        persona_safe,
                        run_number,
                        i,
                        session_type,
                        {persona_safe},
                    )
                    print(
                        f"  Skipping session {i}/{len(session_types_prepared)} "
                        f"({session_type}): transcript already exists"
                    )
                    sessions.append(
                        {
                            "index": i,
                            "session_type": session_type,
                            "turns": 0,
                            "conversation": [],
                            "filename": existing_path,
                            "log_file": None,
                            "duration": 0.0,
                            "early_termination": False,
                            "skipped": True,
                            "skip_reason": "existing",
                        }
                    )
                    continue

                if i > 1:
                    print(
                        f"  Session {i - 1} finished. Starting session "
                        f"{i}/{len(session_types_prepared)}: {session_type}"
                    )
                else:
                    print(
                        f"  Starting session {i}/{len(session_types_prepared)}: "
                        f"{session_type}"
                    )

                await agent.enter_session(session_type)

                first_speaker = agent.first_speaker
                psf = (
                    (first_speaker == Role.PERSONA)
                    if first_speaker is not None
                    else self.persona_speaks_first
                )

                session_stem = (
                    f"{filename_base}_s{i}_{session_type}"
                    if multi_session
                    else filename_base
                )
                session_logger = setup_conversation_logger(
                    log_filename=session_stem, log_dir=self.logs_dir
                )
                logger = session_logger
                log_conversation_start(
                    logger=session_logger,
                    llm1_model_str=model_name,
                    llm1_prompt=persona_name,
                    llm2_name=agent.name,
                    llm2_model_str=getattr(agent, "model_name", "unknown"),
                    max_turns=max_turns,
                    persona_speaks_first=psf,
                    llm1_model=persona,
                    llm2_model=agent,
                )
                session_start = time.time()

                try:
                    conversation = await simulator.generate_conversation(
                        max_turns=max_turns,
                        max_total_words=self.max_total_words,
                        persona_speaks_first=psf,
                    )
                except LLMGenerationFailed as e:
                    end_time = time.time()
                    conversation_time = end_time - start_time
                    print(
                        f"Skipped conversation ({persona_name}, run {run_number}): {e}"
                    )
                    if logger is not None:
                        logger.error(
                            "CONVERSATION FAILED | persona=%s run=%s error=%s",
                            persona_name,
                            run_number,
                            str(e),
                        )
                    cleanup_logger(session_logger)
                    sessions.append(
                        {
                            "index": i,
                            "session_type": session_type,
                            "turns": 0,
                            "conversation": [],
                            "filename": None,
                            "log_file": os.path.join(
                                self.logs_dir, f"{session_stem}.log"
                            ),
                            "duration": time.time() - session_start,
                            "early_termination": False,
                            "skipped": True,
                            "error": str(e),
                        }
                    )
                    result = self._job_result(
                        index=conversation_index,
                        llm1_model=model_name,
                        llm1_prompt=persona_name,
                        run_number=run_number,
                        sessions=sessions,
                        duration=conversation_time,
                        skipped=True,
                        error=str(e),
                    )
                    break
                else:
                    self._log_turns(session_logger, conversation)
                    session_time = time.time() - session_start
                    session_early_term = any(
                        t.get("early_termination", False) for t in conversation
                    )
                    log_conversation_end(
                        logger=session_logger,
                        total_turns=len(conversation),
                        early_termination=session_early_term,
                        total_time=session_time,
                    )

                    output_filename = f"{session_stem}.txt"
                    simulator.save_conversation(output_filename, self.transcripts_dir)
                    sessions.append(
                        {
                            "index": i,
                            "session_type": session_type,
                            "turns": len(conversation),
                            "conversation": conversation,
                            "filename": os.path.join(
                                self.transcripts_dir, output_filename
                            ),
                            "log_file": os.path.join(
                                self.logs_dir, f"{session_stem}.log"
                            ),
                            "duration": session_time,
                            "early_termination": session_early_term,
                            "skipped": False,
                        }
                    )
                    cleanup_logger(session_logger)

            if result is None:
                conversation_time = time.time() - start_time
                result = self._job_result(
                    index=conversation_index,
                    llm1_model=model_name,
                    llm1_prompt=persona_name,
                    run_number=run_number,
                    sessions=sessions,
                    duration=conversation_time,
                    skipped=False,
                )

        except Exception as exc:
            end_time = time.time()
            if logger is not None:
                logger.error(
                    "RUN FAILED | persona=%s run=%s error=%s",
                    persona_name,
                    run_number,
                    str(exc),
                )
            result = self._job_result(
                index=conversation_index,
                llm1_model=model_name,
                llm1_prompt=persona_name,
                run_number=run_number,
                sessions=[],
                duration=end_time - start_time,
                skipped=True,
                skip_reason="error",
                error=str(exc),
            )
            print(f"Skipped conversation ({persona_name}, run {run_number}): {exc}")
        finally:
            if logger is not None:
                cleanup_logger(logger)

            for llm in (persona, agent):
                if llm is not None:
                    try:
                        await llm.cleanup()
                    except Exception as e:
                        print(f"Warning: Failed to cleanup LLM: {e}")

        assert result is not None
        return result

    def _log_turns(self, logger, conversation: List[Dict[str, Any]]) -> None:
        for i, turn in enumerate(conversation, 1):
            log_conversation_turn(
                logger=logger,
                turn_number=i,
                speaker=turn.get("speaker", "Unknown"),
                input_message=turn.get("input", ""),
                response=turn.get("response", ""),
                early_termination=turn.get("early_termination", False),
                logging=turn.get("logging", {}),
            )

    async def run_conversations(
        self, persona_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Run multiple conversations concurrently using queue workers."""
        personas = load_prompts_from_csv(persona_names, max_personas=self.max_personas)
        persona_safe_names = {
            persona_token_for_transcript_stem(p["Name"]) for p in personas
        }
        existing_list = (
            self._list_existing_conversations(persona_safe_names) if self.resume else []
        )
        complete_runs, sessions_by_run = self._build_resume_state(existing_list)

        # Create jobs for all conversations (each prompt run multiple times)
        jobs: List[Tuple[dict, int, int, int]] = []
        skipped_results: List[Dict[str, Any]] = []
        conversation_index = 1

        for persona in personas:
            for run in range(1, self.runs_per_prompt + 1):
                persona_safe = persona_token_for_transcript_stem(persona["Name"])
                if self._has_existing_transcript(
                    persona_safe, run, complete_runs, sessions_by_run
                ):
                    skipped_results.append(
                        self._job_result(
                            index=conversation_index,
                            llm1_model=self.persona_model_config["model"],
                            llm1_prompt=persona["Name"],
                            run_number=run,
                            sessions=[],
                            duration=0.0,
                            skipped=True,
                            skip_reason="existing",
                            error="Transcript already exists in output folder",
                        )
                    )
                    conversation_index += 1
                    continue
                jobs.append(
                    (
                        {
                            "model": self.persona_model_config["model"],
                            "prompt": persona["prompt"],
                            "name": persona["Name"],
                            "run": run,
                        },
                        self.max_turns,
                        conversation_index,
                        run,
                    )
                )
                conversation_index += 1

        total_jobs = len(jobs)
        start_time = datetime.now()
        queue: Queue = Queue()
        for job in jobs:
            await queue.put(job)

        if self.max_concurrent is not None and self.max_concurrent < 0:
            raise ValueError(
                "max_concurrent must be None, 0 (no limit), or a positive integer"
            )

        if total_jobs == 0:
            print("No conversation jobs to run (queue is empty).")
            num_workers = 0
        elif self.max_concurrent in (None, 0):
            num_workers = total_jobs
            print(f"Running {total_jobs} conversations concurrently (no limit)")
        else:
            num_workers = min(self.max_concurrent, total_jobs)
            print(
                f"Running {total_jobs} conversations with max concurrency: "
                f"{self.max_concurrent} ({num_workers} workers)"
            )

        results: List[Dict[str, Any]] = []
        workers = [
            asyncio.create_task(self._worker(i, queue, results, total_jobs))
            for i in range(num_workers)
        ]
        await asyncio.gather(*workers)

        results = skipped_results + results

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        skipped_n = sum(1 for r in results if r.get("skipped"))
        skipped_existing_n = sum(
            1
            for r in results
            if r.get("skipped") and r.get("skip_reason") == "existing"
        )
        skipped_error_n = sum(
            1 for r in results if r.get("skipped") and r.get("skip_reason") == "error"
        )
        print(
            f"\nCompleted {len(results) - skipped_n} / {len(results)} "
            f"conversations in "
            f"{total_time:.2f} seconds"
        )
        if skipped_existing_n:
            print(f"  ({skipped_existing_n} skipped: transcript already exists)")
        if skipped_error_n:
            print(f"  ({skipped_error_n} skipped due to errors)")

        return results
