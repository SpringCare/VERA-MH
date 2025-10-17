# LLM Judge System

This folder contains the LLM Judge system for evaluating conversations using structured rubrics.

## Components

- **`llm_judge.py`** - Core LLMJudge class for evaluation
- **`evaluate_conversations.py`** - Standalone evaluation utilities
- **`rubrics/`** - JSON rubric definitions
- **`__init__.py`** - Package initialization

## Rubrics

### Available Rubrics

1. **`helpfulness.json`** - Evaluates how helpful responses are
   - Relevance: How well responses address user needs
   - Completeness: Whether answers are thorough
   - Actionability: Quality of practical guidance

2. **`safety.json`** - Evaluates safety and ethics
   - Harm Prevention: Avoiding dangerous information
   - Bias and Fairness: Fair treatment and avoiding stereotypes
   - Privacy Respect: Appropriate handling of sensitive information

3. **`communication.json`** - Evaluates communication quality
   - Clarity: How clear and coherent responses are
   - Tone Appropriateness: Suitable tone for context
   - Engagement: Responsiveness to conversational cues

### Rubric Format

```json
{
  "id": "rubric_id",
  "name": "Human Readable Name",
  "description": "Description of what this rubric evaluates",
  "dimensions": {
    "dimension_id": {
      "name": "Dimension Name",
      "description": "What this dimension measures",
      "scale": {
        "min": 1,
        "max": 5
      },
      "criteria": {
        "1": "Description for score 1",
        "5": "Description for score 5"
      }
    }
  }
}
```

## Usage

### Judge Model Selection

The system supports all major LLM providers as judges:

**OpenAI Models:**
- `gpt-4` (default)
- `gpt-4-turbo`
- `gpt-3.5-turbo`

**Claude Models:**
- `claude-3-5-sonnet-20241022`
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`

**Gemini Models:**
- `gemini-1.5-pro`
- `gemini-1.5-flash`
- `gemini-pro`

**Llama Models (via Ollama):**
- `llama3:8b`
- `llama3:70b`
- `llama2:13b`

### Standalone Evaluation

```bash
# Basic evaluation (uses gpt-4 by default)
python evaluate.py -c conversations/file.txt

# Use Claude as judge
python evaluate.py -c conversations/file.txt -j claude-3-5-sonnet-20241022

# Use Gemini as judge
python evaluate.py -c conversations/file.txt -j gemini-1.5-pro

# Use Llama as judge (requires Ollama running locally)
python evaluate.py -c conversations/file.txt -j llama3:8b

# Evaluate all conversations in folder with specific rubrics
python evaluate.py -f conversations/ -r helpfulness safety -j gpt-4-turbo

# From judge folder (advanced usage)
python judge/evaluate_conversations.py -c file.txt -j claude-3-opus-20240229
```

### Integrated with Conversation Generation

```python
from generate_conversations import ConversationRunner

# Using OpenAI as judge
runner = ConversationRunner(
    llm1_model="gpt-4",
    llm2_prompt="philosopher", 
    enable_evaluation=True,
    judge_model="gpt-4",  # Simple line of code!
    rubric_files=["helpfulness", "communication"]
)

# Using Claude as judge  
runner = ConversationRunner(
    llm1_model="gpt-4",
    llm2_prompt="philosopher", 
    enable_evaluation=True,
    judge_model="claude-3-5-sonnet-20241022",  # Simple line of code!
    rubric_files=["helpfulness", "safety"]
)

# Using Gemini as judge
runner = ConversationRunner(
    llm1_model="gpt-4",
    llm2_prompt="philosopher", 
    enable_evaluation=True,
    judge_model="gemini-1.5-pro",  # Simple line of code!
    rubric_files=["communication"]
)

# Using Llama as judge (requires Ollama)
runner = ConversationRunner(
    llm1_model="gpt-4",
    llm2_prompt="philosopher", 
    enable_evaluation=True,
    judge_model="llama3:8b",  # Simple line of code!
    rubric_files=["helpfulness", "safety", "communication"]
)

results = await runner.run_conversations(["assistant"])
```

### Quick Provider Selection

```python
from judge import LLMJudge

# Get default model for each provider
openai_judge = LLMJudge.get_default_model_by_provider("openai")     # "gpt-4"
claude_judge = LLMJudge.get_default_model_by_provider("claude")     # "claude-3-5-sonnet-20241022"
gemini_judge = LLMJudge.get_default_model_by_provider("gemini")     # "gemini-1.5-pro"
llama_judge = LLMJudge.get_default_model_by_provider("llama")       # "llama3:8b"

# List all supported models
all_models = LLMJudge.list_supported_models()
print(all_models)
```

## Output Format

Evaluations are saved as JSON files with this structure:

```json
{
  "conversation_file": "path/to/conversation.txt",
  "judge_model": "gpt-4",
  "evaluations": {
    "helpfulness": {
      "rubric_id": "helpfulness",
      "rubric_name": "Helpfulness Assessment",
      "scores": {
        "relevance": {
          "score": 4,
          "justification": "Response addressed the question directly..."
        }
      }
    }
  }
}
```

## Creating Custom Rubrics

1. Create a new JSON file in `rubrics/` following the format above
2. Include unique `id` and multiple `dimensions`
3. Define scoring `scale` and `criteria` for each dimension
4. Reference the rubric by filename (without .json) when evaluating