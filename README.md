# LLM Conversation Simulator

A Python application that simulates conversations between two Large Language Models (LLMs) using LangChain. The architecture is designed to be extensible, allowing different LLM providers to be easily integrated.

## Features

- **Modular Architecture**: Abstract LLM interface allows for easy integration of different LLM providers
- **System Prompts**: Each LLM instance can be initialized with custom system prompts loaded from files
- **Multiple Prompt Options**: Pre-built prompts for different AI personalities (assistant, philosopher, creative, scientist, skeptic)
- **Early Stopping**: Conversations can end naturally when the first LLM signals completion
- **Conversation Tracking**: Full conversation history is maintained and can be saved to files
- **LangChain Integration**: Uses LangChain for robust LLM interactions
- **Claude Support**: Full implementation of Claude models via Anthropic's API
- **OpenAI Support**: Complete integration with GPT models via OpenAI's API

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your Anthropic API key
   ```

3. **Run the simulation**:
   ```bash
   python main.py
   ```

## Architecture

### Core Components

- **`llm_interface.py`**: Abstract base class defining the LLM interface
- **`llm_factory.py`**: Factory class for creating LLM instances based on model name/version
- **`claude_llm.py`**: Claude implementation using LangChain
- **`conversation_simulator.py`**: Manages conversations between two LLM instances with early stopping support
- **`config.py`**: Configuration management for API keys and model settings for multiple providers
- **`main.py`**: Clean entry point for running simulations
- **`utils/`**: Utility functions and helpers
  - `prompt_loader.py`: Functions for loading prompt files
  - `model_config_loader.py`: Model configuration management
  - `conversation_utils.py`: Conversation formatting and file operations
  - `__init__.py`: Package exports for easy importing
- **`prompts/`**: Directory containing AI personality prompts (system prompt + initial message)
  - `assistant.txt`: Helpful and concise assistant (Claude)
  - `philosopher.txt`: Deep thinker who asks thoughtful questions (Claude)
  - `debate_starter.txt`: Intellectual debater focused on AI and consciousness (Claude)
  - `creative.txt`: Imaginative and unconventional problem solver (Claude)
  - `scientist.txt`: Analytical and evidence-based reasoner (Claude)
  - `skeptic.txt`: Critical thinker who questions assumptions (Claude)
  - `gpt_assistant.txt`: Helpful AI assistant (OpenAI)
  - `gpt_creative.txt`: Creative and innovative thinker (OpenAI)
  - `gpt_analyst.txt`: Structured analytical reasoning (OpenAI)
- **`model_config.json`**: Model assignments for each prompt (separate from prompt content)

### Adding New LLM Providers

To add support for a new LLM provider:

1. Create a new class that inherits from `LLMInterface`
2. Implement the required methods: `generate_response()` and `set_system_prompt()`
3. Update the configuration as needed
4. Use the new LLM class in your simulations

## Usage

The basic usage involves loading prompt configurations and running a conversation:

```python
from llm_factory import LLMFactory
from conversation_simulator import ConversationSimulator
from utils.prompt_loader import load_prompt_config

# Load prompt configurations (model from model_config.json, prompt from prompts/)
config1 = load_prompt_config("assistant")     # Model: claude-3-5-sonnet-20241022
config2 = load_prompt_config("philosopher")   # Model: claude-3-opus-20240229

# Create LLM instances using models from separate configuration
llm1 = LLMFactory.create_llm(
    model_name=config1["model"],
    name="Assistant", 
    system_prompt=config1["system_prompt"]
)

llm2 = LLMFactory.create_llm(
    model_name=config2["model"], 
    name="Philosopher",
    system_prompt=config2["system_prompt"]
)

# Run simulation with initial message from first prompt
simulator = ConversationSimulator(llm1, llm2)
conversation = simulator.start_conversation(config1["initial_message"], max_turns=5)
```

### Supported Models

Currently supported models:
- **Claude**: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`
- **OpenAI**: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`

### Custom Prompts and Models

The system uses **separated configuration** for better maintainability:

#### 1. Create Prompt Files (`prompts/`)
Add `.txt` files containing system prompts and initial messages:

```
You are a helpful AI assistant. Keep your responses concise and informative.

---INITIAL_MESSAGE---
What do you think makes a good conversation?
```

#### 2. Configure Models (`model_config.json`)
Assign models to prompts in the JSON configuration:

```json
{
  "prompt_models": {
    "assistant": "claude-3-5-sonnet-20241022",
    "philosopher": "claude-3-opus-20240229", 
    "gpt_assistant": "gpt-4",
    "gpt_creative": "gpt-4-turbo",
    "new_prompt": "claude-3-haiku-20240307"
  },
  "default_model": "claude-3-5-sonnet-20241022"
}
```

**Benefits of Separation:**
- **Clean Prompts**: Focus on personality and behavior, not technical details
- **Easy Model Changes**: Switch models for existing prompts without touching prompt files
- **Centralized Model Management**: All model assignments in one place
- **Version Control Friendly**: Prompt changes don't require model config changes

### Early Stopping

The conversation simulator supports natural conversation termination when the first LLM (conversation initiator) signals that the conversation is complete.

**Termination Signals Detected:**
- Explicit endings: "goodbye", "bye", "farewell", "conversation over"
- Natural conclusions: "in conclusion", "to conclude", "final thoughts"
- Polite endings: "thanks for", "pleasure talking", "great conversation"
- Direct signals: "i'm done", "let's end here", "nothing more to discuss"

**How It Works:**
1. Only the first LLM (conversation initiator) can trigger early termination
2. When termination signals are detected, the conversation ends immediately
3. The conversation history includes termination flags for analysis
4. Both console output and saved files indicate early termination

**Example:**
```python
# Conversation will end naturally if LLM1 says something like:
# "Thanks for the great discussion! I think we've covered everything. Goodbye!"
# Instead of continuing for the full max_turns
```

## Configuration

Model settings can be adjusted in `config.py`:

- Model name/version
- Temperature
- Max tokens
- Other provider-specific parameters