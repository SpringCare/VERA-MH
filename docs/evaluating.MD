# Evaluate your own model or service

VERA-MH is ready to be used to evaluate any chat-based interface. 
[This](../llm_clients/llm_interface.py) Abstract Base Class (ABC)  represent the interface to be implemebnted. 
Three concrete implementations of that class are provided, for the APIs of ChatGPT, Claude and Gemini.

To test your service, you need to instantiate a concrete class and wrap its endpoint to the `generate_response` function. The function returns a string—the chatbot response—given a string, the user input.

Specifically, to add support for a new LLM provider:

1. Create a new class that inherits from `LLMInterface`
2. Implement the required methods: `generate_response()` and `set_system_prompt()`
3. Update the configuration as needed
4. Add the new LLM client to the [llm_factory.py](../llm_clients/llm_factory.py) file
5. Use the new LLM class in your simulations



The current implementations uses `async` to avoid blocking when the various conversations are being generated. 

Then, assuming all permission have been properly set, you can run it via 
```bash
python3 generate.py <appropriate flags here>
```
 