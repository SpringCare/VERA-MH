from typing import Optional

from .llm_interface import JudgeLLM, LLMInterface


class LLMFactory:
    """Factory class for creating LLM instances based on model name/version."""

    @staticmethod
    def create_llm(
        model_name: str, name: str, system_prompt: Optional[str] = None, **kwargs
    ) -> LLMInterface:
        """
        Create an LLM instance based on the model name.

        This method returns the general LLMInterface type. If you need
        structured output capabilities (e.g., for judge operations), use
        create_judge_llm() or check isinstance(llm, JudgeLLM).

        Args:
            model_name: The model identifier
                (e.g., "claude-3-5-sonnet-20241022", "gpt-4")
            name: Display name for this LLM instance
            system_prompt: Optional system prompt
            **kwargs: Additional model-specific parameters
                (temperature, max_tokens, etc.)

        Returns:
            LLMInterface instance

        Raises:
            ValueError: If model name is not recognized
        """
        # Normalize model name to determine provider
        model_lower = model_name.lower()

        # Filter out non-model-specific parameters
        model_params = {
            k: v
            for k, v in kwargs.items()
            if k not in ["model", "name", "prompt_name", "system_prompt"]
        }

        if "claude" in model_lower:
            from .claude_llm import ClaudeLLM

            return ClaudeLLM(name, system_prompt, model_name, **model_params)
        elif "gpt" in model_lower or "openai" in model_lower:
            from .openai_llm import OpenAILLM

            return OpenAILLM(name, system_prompt, model_name, **model_params)
        elif "gemini" in model_lower or "google" in model_lower:
            from .gemini_llm import GeminiLLM

            return GeminiLLM(name, system_prompt, model_name, **model_params)
        elif "llama" in model_lower or "ollama" in model_lower:
            from .llama_llm import LlamaLLM

            return LlamaLLM(name, system_prompt, model_name, **model_params)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    @staticmethod
    def create_judge_llm(
        model_name: str, name: str, system_prompt: Optional[str] = None, **kwargs
    ) -> JudgeLLM:
        """
        Create an LLM instance with structured output capabilities.

        This method is a type-safe wrapper for creating LLMs that can be used
        as judges. It ensures the returned instance supports structured output.

        Args:
            model_name: The model identifier
                (e.g., "claude-3-5-sonnet-20241022", "gpt-4")
            name: Display name for this LLM instance
            system_prompt: Optional system prompt
            **kwargs: Additional model-specific parameters
                (temperature, max_tokens, etc.)

        Returns:
            JudgeLLM instance with structured output support

        Raises:
            ValueError: If model doesn't support structured output (e.g., Llama/Ollama)
        """
        llm = LLMFactory.create_llm(model_name, name, system_prompt, **kwargs)

        if not isinstance(llm, JudgeLLM):
            raise ValueError(
                f"Model '{model_name}' does not support structured output "
                f"generation. Judge operations require models with structured "
                f"output support. Supported models: Claude (claude-*), "
                f"OpenAI (gpt-*), Gemini (gemini-*). "
                f"Not supported: Llama/Ollama models."
            )

        return llm

    @staticmethod
    def supports_structured_output(model_name: str) -> bool:
        """
        Check if a model supports structured output generation.

        Args:
            model_name: The model identifier to check

        Returns:
            True if model supports structured output, False otherwise
        """
        model_lower = model_name.lower()
        # Llama/Ollama models don't support structured output
        if "llama" in model_lower or "ollama" in model_lower:
            return False
        # All other supported models do
        return True
