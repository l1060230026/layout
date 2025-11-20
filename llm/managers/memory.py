import tiktoken
import json
from langchain.prompts import PromptTemplate

class CustomSummaryMemory:
    def __init__(self, system, llm, max_tokens=2000, keep_recent=2, room_state_llm=None, relationship_llm=None):
        self.system = system
        self.llm = llm
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent  # Keep the most recent N rounds of conversation
        self.raw_buffer = []           # Store complete conversation records
        self.summary = ""              # Store compressed summary
        self.room_state_llm = room_state_llm  # Add room_state_llm reference
        self.relationship_llm = relationship_llm  # Add relationship_llm reference
        
        # Add system message and room information to buffer
        room_state_info = self._get_room_state_info()
        relationship_info = self._get_relationship_info()
        self.raw_buffer.append(f"System: {system}\nCurrent Room State: {room_state_info}\nCurrent Room Relationships: {relationship_info}")
        self.buffer = "\n".join(self.raw_buffer)
        
    def _get_room_state_info(self):
        """Get current room state information"""
        if self.room_state_llm:
            rooms = self.room_state_llm.room_manager.query_rooms()
            if rooms:
                return f"Confirmed rooms: {', '.join([f'{k}({v})' for k, v in rooms.items()])}"
            return "No rooms confirmed yet"
        return "Room state tracking not available"
        
    def _get_relationship_info(self):
        """Get current room relationship information"""
        if self.relationship_llm:
            relationships = self.relationship_llm.relationship_manager.query_relationships()
            if relationships:
                connections = []
                for (room1, room2), has_door in relationships.items():
                    door_status = "with door" if has_door else "without door"
                    connections.append(f"{room1} - {room2} ({door_status})")
                return f"Room connections: {', '.join(connections)}"
            return "No room connections confirmed yet"
        return "Room relationship tracking not available"
        
    def add_user_message(self, message):
        """Add user message"""
        self.raw_buffer.append(f"User: {message}")
        self._summarize_if_needed()
        self.buffer = "\n".join(self.raw_buffer)
        
    def add_ai_message(self, message):
        """Add AI reply message"""
        self.raw_buffer.append(f"AI: {message}")
        self._summarize_if_needed()
        self.buffer = "\n".join(self.raw_buffer)
        
    def _summarize_if_needed(self):
        """Check if summary generation is needed"""
        if self._get_total_tokens() > self.max_tokens:
            self._summarize()
            
    def _summarize(self):
        """Generate conversation summary and compress buffer"""
        # Prompt template for generating summary
        prompt = PromptTemplate(
            template="Please summarize the following dialogue content: \n{dialogue}",
            input_variables=["dialogue"]
        )
        # Keep the most recent N rounds of conversation (user and AI appear in pairs)
        recent_messages = self.raw_buffer[-self.keep_recent*2:]
        # Keep system and room information
        recent_messages.insert(0, self.raw_buffer[0])
        # Historical conversation to be summarized (excluding recently kept and system parts)
        history_messages = self.raw_buffer[1:-self.keep_recent*2]
        
        # Generate summary
        chain = prompt | self.llm
        history_summary = chain.invoke("\n".join(history_messages))
        
        # Update summary (merge old summary and newly generated summary)
        self.summary = f"{self.summary}\n{history_summary}".strip()
        # Reset buffer, keeping only the most recent N rounds of conversation
        self.raw_buffer = recent_messages
        self.buffer = "\n".join(self.raw_buffer)

    def _get_total_tokens(self):
        """Use tiktoken to accurately calculate token count"""
        try:
            # Get corresponding encoder based on model name
            encoding = tiktoken.encoding_for_model(self.llm.model_name)
        except KeyError:
            # If model is not supported, default to cl100k_base
            encoding = tiktoken.get_encoding("cl100k_base")

        # Calculate conversation buffer token count
        buffer_tokens = 0
        for message in self.raw_buffer:
            buffer_tokens += len(encoding.encode(message))
        
        # Calculate summary token count
        summary_tokens = len(encoding.encode(self.summary))
        
        return buffer_tokens + summary_tokens 