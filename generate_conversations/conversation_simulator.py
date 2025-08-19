from typing import List, Dict, Any, Set
from llm_clients import LLMInterface
from utils.conversation_utils import save_conversation_to_file, format_conversation_summary

class ConversationSimulator:
    """Simulates a conversation between two LLM instances."""
    
    def __init__(self, llm1: LLMInterface, llm2: LLMInterface):
        self.llm1 = llm1
        self.llm2 = llm2
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Define termination signals that indicate LLM1 wants to end the conversation
        self.termination_signals: Set[str] = {
            "goodbye", "bye", "farewell", "talk to you later", "ttyl",
            "end conversation", "conversation over", "that's all", 
            "nothing more to discuss", "i'm done", "let's end here",
            "conversation complete", "wrapping up", "final thoughts",
            "concluding", "to conclude", "in conclusion"
        }
    
    def _should_terminate_conversation(self, response: str, speaker: LLMInterface) -> bool:
        """
        Check if the response indicates the conversation should end.
        Only terminates if LLM1 (the conversation initiator) signals to end.
        
        Args:
            response: The response text to check
            speaker: The LLM instance that generated the response
            
        Returns:
            True if conversation should terminate, False otherwise
        """
        # Only allow LLM1 to terminate the conversation early
        if speaker != self.llm1:
            return False
        
        response_lower = response.lower()
        
        # Check for exact phrase matches
        for signal in self.termination_signals:
            if signal in response_lower:
                return True
        
        # Check for common ending patterns
        ending_patterns = [
            "thanks for",
            "thank you for",
            "it was nice",
            "pleasure talking",
            "great conversation",
            "good chat",
            "until next time"
        ]
        
        for pattern in ending_patterns:
            if pattern in response_lower:
                return True
        
        return False
    
    async def start_conversation(self, max_turns: int = 10) -> List[Dict[str, Any]]:
        """
        Start a conversation between the two LLMs with early stopping support.
        
        Args:
            initial_message: The message to start the conversation
            max_turns: Maximum number of conversation turns
            
        Returns:
            List of conversation turns with speaker and message
        """
        self.conversation_history = []
        current_message = ""
        current_speaker = self.llm1
        next_speaker = self.llm2
        
        for turn in range(max_turns):
            response = await current_speaker.generate_response(current_message)
            
            # Record this turn
            self.conversation_history.append({
                "turn": turn + 1,
                "speaker": current_speaker.get_name(),
                "input": current_message,
                "response": response,
                "early_termination": False
            })
            
            # Check if LLM1 wants to end the conversation
            if self._should_terminate_conversation(response, current_speaker):
                self.conversation_history[-1]["early_termination"] = True
                break
            
            # Switch speakers and use the response as the next input
            current_message = response
            current_speaker, next_speaker = next_speaker, current_speaker
        
        return self.conversation_history
    
    # def get_conversation_summary(self) -> str:
    #     """Get a formatted summary of the conversation."""
    #     return format_conversation_summary(self.conversation_history, self.llm1.get_name())
    
    def save_conversation(self, filename: str, folder='conversations') -> None:
        """Save the conversation to a text file."""
        save_conversation_to_file(self.conversation_history, filename, folder, self.llm1.get_name())