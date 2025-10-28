from typing import List, Dict, Any, Set, Optional
from llm_clients import LLMInterface
from utils.conversation_utils import save_conversation_to_file
import time

class ConversationSimulator:
    """Simulates a conversation between two LLM instances."""
    
    def __init__(self, persona: LLMInterface, agent: LLMInterface):
        self.persona = persona
        self.agent = agent
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Define termination signals that indicate persona wants to end the conversation
        self.termination_signals: Set[str] = set()
        
            # "goodbye", "bye", "farewell", "talk to you later", 
            # "ttyl",
            # "end conversation", "conversation over", "that's all", 
            # "nothing more to discuss", 
            # "i'm done", "let's end here",
            # "conversation complete", "wrapping up", "final thoughts",
            # "concluding", "to conclude", 
            # "in conclusion"
        # }
    
    def _should_terminate_conversation(self, response: str, speaker: LLMInterface) -> bool:
        """
        Check if the response indicates the conversation should end.
        Only terminates if persona (the conversation initiator) signals to end.
        """
        # Only allow persona to terminate the conversation early
        if speaker != self.persona:
            return False
        
        response_lower = response.lower()
        
        # Check for exact phrase matches
        for signal in self.termination_signals:
            if signal in response_lower:
                return True
        
        # Check for common ending patterns
        ending_patterns = [
            # "it was nice",
            # "pleasure talking",
            # "great conversation",
            # "good chat",
            # "until next time"
        ]
        
        for pattern in ending_patterns:
            if pattern in response_lower:
                return True
        
        return False
    
    async def start_conversation(self, initial_message: Optional[str] = None, max_turns: int = 10) -> List[Dict[str, Any]]:
        """
        Start a conversation between the two LLMs with early stopping support.
        
        Args:
            max_turns: Maximum number of conversation turns
            
        Returns:
            List of conversation turns with speaker and message
        """
        self.conversation_history = []
        if initial_message is None:
            current_message = 'Start the conversation based on the system prompt'
        else:
            current_message = initial_message
        current_speaker = self.persona
        next_speaker = self.agent
        
        for turn in range(max_turns):
            # Record start time for this turn
            turn_start_time = time.time()
            
            # Generate response
            response = await current_speaker.generate_response(current_message)
            
            # Calculate turn duration
            # turn_duration = time.time() - turn_start_time
            
            # Create comprehensive logging information
            # logging_info = {
            #     "timestamp": datetime.now().isoformat(),
            #     "turn_duration_seconds": round(turn_duration, 3),
            #     "model_name": getattr(current_speaker, 'model_name', 'unknown'),
            #     "provider": self._get_provider_type(current_speaker),
            #     "input_length": len(current_message) if current_message else 0,
            #     "output_length": len(response),
            #     "response_id": None,  # Will be populated by individual providers
            #     "usage": {},  # Will be populated by individual providers
            #     "metadata": {}  # Will be populated by individual providers
            # }
            
            # Record this turn
            self.conversation_history.append({
                "turn": turn + 1,
                "speaker": current_speaker.get_name(),
                "input": current_message or "",
                "response": response,
                "early_termination": False,
                "logging": current_speaker.get_last_response_metadata()
            })
            
            # Check if persona wants to end the conversation
            if self._should_terminate_conversation(response, current_speaker):
                self.conversation_history[-1]["early_termination"] = True
                break
            
            # Switch speakers and use the response as the next input
            current_message = response
            current_speaker, next_speaker = next_speaker, current_speaker
        
        return self.conversation_history
    
    def save_conversation(self, filename: str, folder='conversations') -> None:
        """Save the conversation to a text file."""

        # TODO: why is this two functions
        save_conversation_to_file(self.conversation_history, filename, folder, self.persona.get_name())
