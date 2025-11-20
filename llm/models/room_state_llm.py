import json

class RoomStateLLM:
    def __init__(self, room_manager, llm):
        self.room_manager = room_manager
        self.llm = llm
        self.system_prompt = """You are a room state maintenance assistant. Your task is to:
1. Monitor and track changes in room requirements from conversations
2. Update the room state manager accordingly
3. Keep track of room types and quantities

Available room types are: living room, kitchen, bedroom, bathroom, balcony, entrance, dining room, storage.

When you detect changes in room requirements, you should:
- Add new rooms when they are mentioned
- Update quantities when they change
- Remove rooms when they are no longer needed

Always respond with a JSON array containing one or more operation objects. Each operation object should have:
{
    "action": "add|update|delete|none",
    "room_type": "room type if applicable",
    "quantity": number if applicable,
    "reason": "explanation of the change"
}

Example response for multiple operations:
[
    {
        "action": "add",
        "room_type": "bedroom",
        "quantity": 2,
        "reason": "User requested two bedrooms"
    },
    {
        "action": "update",
        "room_type": "bathroom",
        "quantity": 1,
        "reason": "User specified one bathroom"
    }
]

IMPORTANT: Your response must be a valid JSON array. Do not include any markdown code block markers (```json or ```)."""

    def _clean_json_response(self, response: str) -> str:
        """Clean up the response by removing markdown code block markers"""
        cleaned = response.replace('```json', '').replace('```', '').strip()
        return cleaned

    def process_conversation(self, conversation: str) -> list:
        """Process the conversation and update room state if needed"""
        prompt = f"""{self.system_prompt}
Current room state: {self.room_manager.query_rooms()}
Conversation: {conversation}"""

        response = self.llm.invoke(prompt)
        try:
            # Clean up the response before parsing
            cleaned_response = self._clean_json_response(response.content)
            operations = json.loads(cleaned_response)
            if not isinstance(operations, list):
                raise ValueError("Response is not a JSON array")
                
            results = []
            for operation in operations:
                if not isinstance(operation, dict):
                    continue
                    
                if operation["action"] != "none":
                    if operation["action"] == "add":
                        self.room_manager.add_room(operation["room_type"], operation["quantity"])
                    elif operation["action"] == "update":
                        self.room_manager.update_room(operation["room_type"], operation["quantity"])
                    elif operation["action"] == "delete":
                        self.room_manager.delete_room(operation["room_type"])
                results.append(operation)
            return results
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            return [{"action": "none", "reason": "Invalid JSON response"}]
        except Exception as e:
            print(f"Error processing room state update: {e}")
            return [{"action": "none", "reason": "Error processing update"}] 