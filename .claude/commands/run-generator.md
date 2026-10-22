Help the user run the conversation generator (`uv run python vera.py generate`; flags in docs/cli.md):

1. Ask the user for parameters:
   - Chatbot (`-c`) and user (`-u`) models: required, no default (recommended: `llm_clients/config.py`)
   - Number of conversations (default: 5)
   - Any other relevant options
2. Run the command with the specified parameters
3. Show the output and report success/errors
