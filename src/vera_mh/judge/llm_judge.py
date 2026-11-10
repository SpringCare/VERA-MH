"""LLM Judge for evaluating conversations based on rubrics."""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from vera_mh.judge.answers import conversation_answers_dir, write_conversation_answers
from vera_mh.judge.rubric_config import ConversationData, RubricConfig, answer_options
from vera_mh.judge.scoring import score
from vera_mh.judge.utils import (
    default_adhoc_parent,
    judge_evaluation_tsv_filename,
    log_llm_failure_metadata,
    log_llm_response_metadata,
)
from vera_mh.llm_clients import LLMFactory
from vera_mh.llm_clients.llm_interface import JudgeLLM, LLMGenerationFailed


class LLMJudge:
    """Evaluates conversations using LLM-based scoring with rubrics."""

    def __init__(
        self,
        *,
        judge_model: str,
        rubric_config: RubricConfig,
        judge_model_extra_params: Optional[Dict[str, Any]] = None,
        log_file: Optional[str] = None,
    ):
        """
        Initialize the LLM Judge with pre-loaded rubric configuration.

        Args:
            judge_model: Model to use for judging
            rubric_config: Pre-loaded rubric configuration data
            judge_model_extra_params: Extra parameters for the judge model
            log_file: Path to log file (default under <adhoc>/judge_unscoped/; see
                default_adhoc_parent() / VERA_ADHOC_PARENT)
        """

        # Setup logger: batch judging passes an explicit path from
        # build_judge_task_log_path (scoped to the run). Omitted log_file is only
        # for direct construction, tests, etc.—then judge_unscoped/ under
        # default_adhoc_parent(); UUID stem avoids concurrent overwrites.
        scoped = log_file is not None
        if not scoped:
            log_file = str(
                Path(default_adhoc_parent())
                / "judge_unscoped"
                / f"judge_{uuid.uuid4().hex}.log"
            )

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"LLMJudge_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # One log path per batch job: stem matches the evaluation TSV (see
        # build_judge_task_log_path). Different evaluation runs use different
        # output folders, so logs do not collide across runs.
        # mode="w" truncates on each job start; re-running the same job in the
        # same folder replaces the log (and TSV), which is intentional.
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.log_file = log_file
        self.judge_model = judge_model
        self.judge_model_extra_params = judge_model_extra_params or {}

        # Set default temperature to 0 for more deterministic judge behavior
        # Users can override this by passing temperature in judge_model_extra_params
        if "temperature" not in self.judge_model_extra_params:
            self.judge_model_extra_params["temperature"] = 0

        # The rubric: its questions and routes (`walk`), and its prompts
        self.rubric = rubric_config

        # Initialize evaluator (created per conversation evaluation)
        self.evaluator: Optional[JudgeLLM] = None

        # Log initialization info
        self.logger.info("=== Initializing LLM Judge ===")
        self.logger.info(f"Judge model: {judge_model}")
        self.logger.info(f"Log file: {log_file}")
        questions = len(rubric_config.question_flow_data)
        self.logger.info(f"Loaded question-flow rubric with {questions} questions")
        self.logger.info(f"Loaded dimensions: {rubric_config.dimensions}")
        print(f"Loaded question-flow rubric with {questions} questions")

    def _create_evaluator(
        self, conversation: str, conversation_filename: str
    ) -> JudgeLLM:
        """Create and configure the LLM evaluator with conversation context.

        Args:
            conversation: Conversation text content
            conversation_filename: Filename for logging purposes

        Returns:
            JudgeLLM instance configured for evaluation

        Raises:
            ValueError: If the judge model doesn't support structured output
        """
        # Log evaluation start
        self.logger.info("=" * 80)
        self.logger.info(f"Starting evaluation: {conversation_filename}")
        self.logger.info(f"Model: {self.judge_model}")
        self.logger.info("=" * 80)
        conv_preview = conversation[:1000]
        self.logger.info(
            f"CONVERSATION (length: {len(conversation)} chars):\n{conv_preview}..."
        )

        # Create conversation context prompt
        conversation_context_prompt = self.rubric.rubric_prompt_beginning.format(
            conversation=conversation
        )
        self.logger.info(f"SYSTEM PROMPT:\n{conversation_context_prompt[:500]}...")

        # Create LLM instance
        llm = LLMFactory.create_llm(
            model_name=self.judge_model,
            name="Question Flow Evaluator",
            role=None,
            system_prompt=conversation_context_prompt,
            **self.judge_model_extra_params,
        )

        # Validate that the LLM supports structured output
        if not isinstance(llm, JudgeLLM):
            raise ValueError(
                f"Model '{self.judge_model}' does not support structured "
                f"output generation. Judge operations require models with "
                f"structured output support. Supported models: "
                f"Claude (claude-*), OpenAI (gpt-*), "
                "Gemini (gemini-*), Azure (azure-*). "
                f"Not supported: Ollama models."
            )

        return llm

    def _close_log_handlers(self) -> None:
        """Close and remove file handlers so large batches do not leak FDs."""
        for handler in list(self.logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
            self.logger.removeHandler(handler)

    async def evaluate_conversation_question_flow(
        self,
        conversation: ConversationData,
        *,
        output_folder: str,
        auto_save: bool = True,
        start_question_id: Optional[str] = None,
        reasoning_length: Optional[int] = None,
        judge_instance: Optional[int] = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        Evaluate conversation using question-flow rubric.

        Main evaluation flow:
        1. Navigate through questions until END/ASSIGN_END or completion
        2. Calculate dimension scores from collected answers
        3. Save results if requested

        Args:
            conversation: ConversationData with content and metadata
            output_folder: Folder to save evaluation results
            auto_save: Whether to automatically save results to files
            start_question_id: Question ID to start with (default: first
                question in rubric)
            reasoning_length: Maximum length of the reasoning to log (default: None)
            judge_instance: Instance number for this judge (for unique filenames)

        Returns:
            Dictionary with dimension names as keys and evaluation results as values
            Format: {dimension: {"score": str, "reasoning": str, ...}}
        """
        # Create evaluator with conversation content
        conversation_filename = conversation.metadata.get("filename", "unknown")
        self.evaluator = self._create_evaluator(
            conversation.content, conversation_filename
        )

        try:
            # Step 1: Walk the rubric, asking each question it routes to
            walk = await self.rubric.walk(self._ask_single_question, start_question_id)

            # Step 2: Rate each dimension from the answers
            results = score(self.rubric, walk)

            # Step 3: Log and save results
            self._log_final_results(results)
            if auto_save:
                self._save_results(
                    conversation,
                    output_folder,
                    results,
                    judge_instance,
                    walk.question_log,
                )

            return results
        finally:
            # Cleanup LLM resources (e.g., close HTTP sessions for Azure)
            if self.evaluator is not None:
                try:
                    await self.evaluator.cleanup()
                except Exception as e:
                    # Log but don't fail if cleanup fails
                    self.logger.warning(f"Failed to cleanup evaluator LLM: {e}")
                self.evaluator = None
            self._close_log_handlers()

    def _save_iterative_evaluation(
        self, results: Dict[str, Dict[str, str]], output_file: Path, sep: str = "\t"
    ):
        """Save iterative evaluation results to TSV file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            # Write header
            f.write(f"Dimension{sep}Score{sep}Reasoning\n")

            # Write each dimension's results
            for dimension, result in results.items():
                score = result["score"]
                reasoning = result["reasoning"].replace("\n", " ").replace("\t", " ")
                f.write(f"{dimension}{sep}{score}{sep}{reasoning}\n")

    def _log_final_results(
        self, results: Dict[str, Dict[str, str]], reasoning_length: Optional[int] = None
    ):
        """Log final evaluation results."""
        self.logger.info("=" * 80)
        self.logger.info("FINAL RESULTS:")
        for dimension, result in results.items():
            self.logger.info(f"{dimension}: {result['score']}")
            if result.get("yes_question_id"):
                reasoning = result["yes_reasoning"][:reasoning_length]
                qid = result["yes_question_id"]
                self.logger.info(f"  Yes at Q{qid}: {reasoning}")
        self.logger.info("=" * 80)

    def _save_results(
        self,
        conversation: ConversationData,
        output_folder: str,
        results: Dict[str, Dict[str, str]],
        judge_instance: Optional[int] = None,
        question_log: Optional[List[Dict[str, Any]]] = None,
    ):
        """Save evaluation results to file.

        Args:
            conversation: ConversationData with metadata
            output_folder: Folder to save results
            results: Evaluation results dictionary
            judge_instance: Optional judge instance number for filename
            question_log: Questions asked, in visit order. When given, the
                per-question answers are written alongside the evaluation as
                answers/by_question/<tsv stem>.tsv
        """
        filename = conversation.metadata.get("filename", "unknown.txt")
        tsv_name = judge_evaluation_tsv_filename(
            filename, self.judge_model, judge_instance
        )
        output_file = Path(output_folder) / tsv_name
        self._save_iterative_evaluation(results, output_file)
        self.logger.info(f"Results saved to: {output_file}")

        if question_log is not None:
            answers_file = conversation_answers_dir(output_folder) / tsv_name
            write_conversation_answers(answers_file, question_log)
            self.logger.info(f"Per-question answers saved to: {answers_file}")

    def _log_structured_output_failure(
        self, question_id: str, exc: LLMGenerationFailed
    ) -> None:
        """Log structured-output failure details to the per-conversation judge log."""
        log_llm_failure_metadata(
            self.logger,
            self.evaluator,
            context=f"question {question_id}",
            error=exc,
        )

    async def _ask_single_question(
        self, question_id: str, question_data: Dict[str, Any]
    ) -> tuple[str, str]:
        """Ask one question with the judge LLM, logging the prompt and reply.

        Returns:
            Tuple of (answer_text, reasoning_text)
        """
        prompt = self.rubric.question_prompt(question_data)
        self.logger.info(f"--- Question {question_id} ---")
        self.logger.info(f"PROMPT:\n{prompt}")

        assert self.evaluator is not None, (
            "Evaluator must be initialized before asking questions"
        )
        options = answer_options(question_data)
        try:
            reply = await self.evaluator.answer(prompt, options)
        except LLMGenerationFailed as exc:
            self._log_structured_output_failure(question_id, exc)
            raise

        self.logger.info(
            f"STRUCTURED RESPONSE:\nanswer='{reply.reply}' "
            f"reasoning='{reply.reasoning}'"
        )
        log_llm_response_metadata(
            self.logger,
            self.evaluator,
            context=f"question {question_id}",
        )
        if not reply.matched:
            self.logger.warning(
                f"Answer '{reply.reply}' does not match valid options "
                f"{options}. Using first option as fallback."
            )
        self.logger.info(f"MATCHED ANSWER: {reply.answer}")
        return reply.answer, reply.reasoning
