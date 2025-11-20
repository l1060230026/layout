import json

class PhaseTransitionLLM:
    def __init__(self, llm, room_state_llm, relationship_llm):
        self.llm = llm
        self.room_state_llm = room_state_llm
        self.relationship_llm = relationship_llm
        self.system_prompt = """You are a phase transition assistant for a floor plan design system. Your task is to analyze the current state and conversation to determine if the system should move to the next phase.

The design process has three phases:
1. Room Confirmation Phase
2. Room Relationship Confirmation Phase
3. Room Layout Diagram Generation Phase

For each phase, you should consider:
1. Whether the current phase's requirements are sufficiently met
2. Whether the user has EXPLICITLY confirmed or approved the current phase's results
3. Whether the AI has EXPLICITLY indicated moving to the next phase
4. Whether there are any critical missing elements that should be addressed first

IMPORTANT RULES:
1. For Phase 1 (Room Confirmation):
   - Transition if the user explicitly confirms the room list
   - Transition if the AI explicitly indicates moving to the next phase
   - Look for explicit confirmation phrases like:
     * "yes", "ok", "that works", "confirmed", "approved"
     * "let's continue", "let's move on", "let's proceed"
     * "I agree", "sounds good", "perfect"
     * "that's fine", "that's good", "that's perfect"
   - Look for AI transition phrases like:
     * "Now, let's move on to the Room Relationship Confirmation Phase"
     * "We'll proceed to the next phase"
     * "Let's continue with the room relationships"
   - If the user only lists rooms or asks questions, DO NOT transition
   - If the AI suggests rooms but user hasn't confirmed, DO NOT transition
   - If the user's response is ambiguous or could be interpreted as a question, DO NOT transition
   - If the user's response contains any form of uncertainty (e.g., "maybe", "I think", "probably"), DO NOT transition

2. For Phase 2 (Room Relationships):
   - Transition if all rooms have been connected and the user explicitly confirms
   - Transition if the AI explicitly indicates moving to the next phase
   - Look for explicit confirmation of relationships
   - Look for AI transition phrases like:
     * "Now, we'll move on to the Room Layout Diagram Generation Phase"
     * "Let's proceed to generate the layout diagram"
     * "We'll now generate the floor plan layout"
   - If the user's response is ambiguous or could be interpreted as a question, DO NOT transition

3. For Phase 3 (Layout Generation):
   - Transition if all relationships are confirmed
   - Transition if the user explicitly requests layout generation
   - Transition if the AI explicitly indicates moving to the next phase
   - If the user's response is ambiguous or could be interpreted as a question, DO NOT transition

EXAMPLES OF VALID CONFIRMATIONS:
- "ok, let's continue" -> VALID
- "yes, that works" -> VALID
- "perfect, let's move on" -> VALID
- "that's good, let's proceed" -> VALID
- "I agree with the room list" -> VALID
- "Now, we'll move on to the next phase" -> VALID (from AI)
- "Let's proceed to generate the layout diagram" -> VALID (from AI)

EXAMPLES OF INVALID CONFIRMATIONS:
- "what about adding a study room?" -> INVALID (question)
- "maybe we should add a balcony" -> INVALID (uncertainty)
- "I think that's good" -> INVALID (uncertainty)
- "probably that's enough" -> INVALID (uncertainty)
- "let me think about it" -> INVALID (uncertainty)

Always respond with a JSON object containing:
{
    "should_transition": true/false,
    "reason": "detailed explanation of your decision",
    "missing_elements": ["list of critical missing elements if any"]
}

Example responses:

For Phase 1 (Room Confirmation):
{
    "should_transition": false,
    "reason": "User has not explicitly confirmed the room list yet",
    "missing_elements": ["User confirmation of room list"]
}

{
    "should_transition": true,
    "reason": "AI explicitly indicated moving to the next phase with 'Now, let's move on to the Room Relationship Confirmation Phase'",
    "missing_elements": []
}

IMPORTANT: Your response must be a valid JSON object. Do not include any markdown code block markers (```json or ```)."""

    def _clean_json_response(self, response: str) -> str:
        """Clean up the response by removing markdown code block markers"""
        cleaned = response.replace('```json', '').replace('```', '').strip()
        return cleaned

    def should_transition(self, current_phase: int, conversation: str) -> dict:
        """Determine if the system should transition to the next phase"""
        # Get current state information
        room_state = self.room_state_llm.room_manager.query_rooms()
        relationships = self.relationship_llm.relationship_manager.query_relationships()
        
        prompt = f"""{self.system_prompt}

Current Phase: {current_phase}
Current Room State: {room_state}
Current Room Relationships: {relationships}
Conversation: {conversation}"""

        response = self.llm.invoke(prompt)
        try:
            cleaned_response = self._clean_json_response(response.content)
            result = json.loads(cleaned_response)
            return result
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            return {"should_transition": False, "reason": "Error processing transition decision", "missing_elements": []}
        except Exception as e:
            print(f"Error processing phase transition: {e}")
            return {"should_transition": False, "reason": "Error processing transition decision", "missing_elements": []} 