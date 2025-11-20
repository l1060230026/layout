import json
from .room_graph import RoomGraphVisualizer

class RoomRelationshipLLM:
    def __init__(self, relationship_manager, llm):
        self.relationship_manager = relationship_manager
        self.llm = llm
        self.graph_visualizer = RoomGraphVisualizer(relationship_manager)  # Initialize graph visualizer
        self.standard_room_types = {
            "living room": ["living room", "living", "lounge", "sitting room"],
            "kitchen": ["kitchen", "cooking area"],
            "bedroom": ["bedroom", "master bedroom", "child's bedroom", "guest bedroom", "main bedroom"],
            "bathroom": ["bathroom", "washroom", "toilet"],
            "balcony": ["balcony", "terrace"],
            "entrance": ["entrance", "hallway", "foyer", "entry"],
            "dining room": ["dining room", "dining area"],
            "storage": ["storage", "storage room", "closet", "wardrobe"]
        }
        self.system_prompt = """You are a room relationship maintenance assistant. Your task is to:
1. Monitor and track changes in room connections from conversations
2. Update the relationship manager accordingly
3. Keep track of which rooms are connected by doors

IMPORTANT RULES:
1. You must ONLY use standard room names from this list:
   - living room
   - kitchen
   - bedroom
   - bathroom
   - balcony
   - entrance
   - dining room
   - storage

2. When multiple rooms of the same type exist (e.g., two bedrooms), you must:
   - Use the format "room_type_number" (e.g., "bedroom_1", "bedroom_2")
   - Map user descriptions like "master bedroom" to the appropriate numbered room
   - Ask for clarification if the mapping is unclear

3. You should process room relationships when:
   - The conversation explicitly mentions connections between rooms
   - The conversation explicitly mentions doors or access between rooms
   - The conversation is in the room relationship confirmation phase
   - The AI suggests specific room connections with clear door specifications

4. For suggested connections:
   - If the AI suggests connections with clear room names and door specifications, treat them as valid connections
   - Each suggested connection must specify:
     a) The exact names of the two rooms being connected
     b) Whether there is a door between them (true/false)
   - Example valid suggestions:
     * "bedroom_1 connects to bathroom_1 with a door" -> VALID
     * "living room connects to entrance with a door" -> VALID
     * "kitchen connects to dining room without a door" -> VALID
   - Example invalid suggestions:
     * "bedroom connects to bathroom" -> INVALID (missing door specification)
     * "living room is accessible from entrance" -> INVALID (vague connection)

5. When processing AI suggestions:
   - Extract each connection statement from the AI's response
   - Convert each connection into a valid operation
   - If a connection is mentioned multiple times, use the most recent specification
   - If a connection is unclear or missing door specification, skip it
   - If a connection uses non-standard room names, standardize them first

When you detect changes in room relationships, you should:
- Add new relationships when they are explicitly mentioned or clearly suggested
- Update door connections when they are explicitly changed
- Remove relationships when they are explicitly removed

Always respond with a JSON array containing one or more operation objects. Each operation object should have:
{
    "action": "add|update|delete|none",
    "room1": "standard_room_name_1",
    "room2": "standard_room_name_2",
    "has_door": true/false if applicable,
    "reason": "explanation of the change"
}

Example response for room relationship changes:
[
    {
        "action": "add",
        "room1": "living room",
        "room2": "kitchen",
        "has_door": true,
        "reason": "AI suggested a door between living room and kitchen"
    }
]

Example response when no room relationships are mentioned:
[
    {
        "action": "none",
        "reason": "Conversation only discusses room types and quantities"
    }
]

IMPORTANT: Your response must be a valid JSON array. Do not include any markdown code block markers (```json or ```)."""

    def _standardize_room_name(self, room_name: str) -> str:
        """Convert any room name to its standard form"""
        room_name = room_name.lower().strip()
        for standard_name, variations in self.standard_room_types.items():
            if room_name in variations or room_name == standard_name:
                return standard_name
        return room_name  # Return as is if no match found

    def _clean_json_response(self, response: str) -> str:
        """Clean up the response by removing markdown code block markers"""
        cleaned = response.replace('```json', '').replace('```', '').strip()
        return cleaned

    def process_conversation(self, conversation: str) -> tuple:
        """Process the conversation and update room relationships if needed"""
        prompt = f"""{self.system_prompt}

Current room relationships: {self.relationship_manager.query_relationships()}
Conversation: {conversation}"""

        response = self.llm.invoke(prompt)
        try:
            cleaned_response = self._clean_json_response(response.content)
            operations = json.loads(cleaned_response)
            if not isinstance(operations, list):
                raise ValueError("Response is not a JSON array")
                
            results = []
            relationships_updated = False
            for operation in operations:
                if not isinstance(operation, dict):
                    continue
                    
                if operation["action"] != "none":
                    # Standardize room names before processing
                    room1 = operation["room1"]
                    room2 = operation["room2"]
                    
                    if operation["action"] == "add":
                        self.relationship_manager.add_relationship(
                            room1, room2, operation["has_door"]
                        )
                        relationships_updated = True
                    elif operation["action"] == "update":
                        self.relationship_manager.update_relationship(
                            room1, room2, operation["has_door"]
                        )
                        relationships_updated = True
                    elif operation["action"] == "delete":
                        self.relationship_manager.delete_relationship(
                            room1, room2
                        )
                        relationships_updated = True
                results.append(operation)
            
            # Generate visualization if relationships were updated
            visualization = None
            if relationships_updated and len(self.relationship_manager.query_relationships()) > 0:
                visualization = self.graph_visualizer.visualize_relationships()
                
            return results, visualization
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            return [{"action": "none", "reason": "Invalid JSON response"}], None
        except Exception as e:
            print(f"Error processing room relationship update: {e}")
            return [{"action": "none", "reason": "Error processing update"}], None 