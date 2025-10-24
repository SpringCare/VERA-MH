# Evaluate your own model or service

VERA-MH is ready to be used to evaluate any chat-based interface. 
[This](../llm_clients/llm_interface.py) Abstract Base Class (ABC)  represent the interface to be implemebnted. 
Three concrete implementations of that class are provided, for the APIs of ChatGPT, Claude and Gemini.

To test your service, you need to instantiate a concrete class and wrap its endpoint to the `generate_response` function. The function returns a strin—the chatbot response—given a string, the user input.

 The current implementations uses `async` to avoid blocking when the various conversations. 

Then, assuming all permission have been properly set, you can run it via ```python3 generate.py```