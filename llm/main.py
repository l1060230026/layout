import gradio as gr
from gradio.components import ChatMessage
import tiktoken
import pandas as pd
import openai
import os
import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np
from PIL import Image
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = ""


room_id = {"living room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "balcony": 5, "entrance": 6, "dining room": 7, "study room": 8,
            "storage": 10 , "front door": 15, "interior_door": 17}

from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from typing import List
from langchain.tools import BaseTool
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
system_message = """
You are an intelligent floor plan designer who strictly follows naming conventions and room relationship rules.

The floor plan design process is divided into three phases:

1. Room Confirmation Phase:
   - Only use these exact room types: living room, kitchen, bedroom, bathroom, balcony, entrance, dining room, storage
   - For multiple rooms of the same type, use numbered format: bedroom_1, bedroom_2, bathroom_1, etc.
   - DO NOT create or reference rooms that haven't been confirmed in this phase
   - DO NOT use descriptive terms like "master", "guest", "second", etc.

2. Room Relationship Confirmation Phase:
   - Only establish connections between rooms that were confirmed in Phase 1
   - Use exact room names (with numbers if multiple) when describing connections
   - Each connection must specify:
     a) The exact names of the two rooms being connected
     b) Whether there is a door between them (true/false)
   - NEVER use vague terms like "accessible from" or "connected to hallway"
   - NEVER reference non-existent or unconfirmed rooms
   - Example correct format: "bedroom_1 connects to bathroom_1 with a door"
   - IMPORTANT: Ensure all rooms are connected to at least one other room - no room should be completely isolated
   - The floor plan must form a connected graph where every room is reachable from any other room

3. Room Layout Diagram Generation Phase:
   - In this phase, an external tool will automatically generate the layout diagram based on the confirmed rooms and relationships
   - You should not attempt to describe or explain the layout
   - Simply wait for the external tool to complete the diagram generation and return it to the user
   - Your response should be minimal, only acknowledging that the layout is being generated

IMPORTANT RULES:
1. Always use exact room names from the allowed list
2. For multiple rooms of same type, always use number suffix (e.g., bedroom_1)
3. Never create or reference rooms that haven't been explicitly confirmed
4. Never use descriptive or relative terms for rooms
5. All room relationships must be explicit and between confirmed rooms

In the following communication, guide the user through these steps while strictly adhering to these naming and relationship rules.
"""

api_key = ""
base_url = ""

model = init_chat_model("gpt-4o", api_key=api_key, base_url=base_url)

# 1. Custom room state management tool
class RoomStateManager():
    def __init__(self):
        self.rooms = {}  # Format: {room_name: room_info}
        self.name_mapping = {}  # Format: {unnumbered_name: numbered_name}

    def _standardize_room_name(self, room_type: str, index: int) -> str:
        """Standardize room name to always include a number suffix"""
        return f"{room_type}_{index}"

    def _get_unnumbered_name(self, room_type: str) -> str:
        """Get the unnumbered version of a room name"""
        return room_type

    def _run(self, action: str, room_type: str = None, quantity: int = None):
        if action == "add":
            for i in range(quantity):
                room_name = self._standardize_room_name(room_type, i+1)
                self.rooms[room_name] = {
                    "type": room_type,
                    "name": room_name
                }
                # For single rooms, also store the unnumbered name mapping
                if quantity == 1:
                    unnumbered_name = self._get_unnumbered_name(room_type)
                    self.name_mapping[unnumbered_name] = room_name
        elif action == "delete":
            # Delete all rooms of a specific type
            rooms_to_delete = [name for name in self.rooms if name.startswith(room_type)]
            for room_name in rooms_to_delete:
                del self.rooms[room_name]
            # Also remove from name mapping
            unnumbered_name = self._get_unnumbered_name(room_type)
            if unnumbered_name in self.name_mapping:
                del self.name_mapping[unnumbered_name]
        elif action == "update":
            # First, count existing rooms of this type
            existing_rooms = [name for name in self.rooms if name.startswith(room_type)]
            current_count = len(existing_rooms)
            
            if quantity > current_count:
                # Add more rooms
                for i in range(current_count + 1, quantity + 1):
                    room_name = self._standardize_room_name(room_type, i)
                    self.rooms[room_name] = {
                        "type": room_type,
                        "name": room_name
                    }
            elif quantity < current_count:
                # Remove excess rooms, starting from the highest number
                for i in range(current_count, quantity, -1):
                    room_name = self._standardize_room_name(room_type, i)
                    if room_name in self.rooms:
                        del self.rooms[room_name]
            
            # Update name mapping for single rooms
            unnumbered_name = self._get_unnumbered_name(room_type)
            if quantity == 1:
                room_name = self._standardize_room_name(room_type, 1)
                self.name_mapping[unnumbered_name] = room_name
            else:
                if unnumbered_name in self.name_mapping:
                    del self.name_mapping[unnumbered_name]
                
        return self.rooms

    def get_room_by_name(self, room_name: str):
        """Get room information by its name, supporting both numbered and unnumbered names"""
        # First try direct lookup
        if room_name in self.rooms:
            return self.rooms[room_name]
        # Then try name mapping for unnumbered names
        if room_name in self.name_mapping:
            return self.rooms[self.name_mapping[room_name]]
        return None

    def query_rooms(self):
        """Return all rooms with their information"""
        return self.rooms

    def add_room(self, room_type: str, quantity: int):
        return self._run("add", room_type, quantity)

    def delete_room(self, room_type: str):
        return self._run("delete", room_type)

    def update_room(self, room_type: str, quantity: int):
        return self._run("update", room_type, quantity)

class RoomStateLLM:
    def __init__(self, room_manager: RoomStateManager, llm):
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
        # Remove ```json and ``` markers
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

# Instantiate collector
room_tool = RoomStateManager()
room_state_llm = RoomStateLLM(room_tool, model)

class RoomRelationshipManager:
    def __init__(self, room_state_manager: RoomStateManager = None):
        self.relationships = {}  # Dictionary to store room connections
        # Format: {(room1, room2): has_door, ...}
        self.room_state_manager = room_state_manager
        self.door_connections = {}  # Dictionary to store door connections
        # Format: {door_id: (room1, room2)}

    def _get_standardized_room_name(self, room_name: str) -> str:
        """Get the standardized room name (with number) from either format"""
        if not self.room_state_manager:
            return room_name
        # First try direct lookup
        if room_name in self.room_state_manager.rooms:
            return room_name
        # Then try name mapping for unnumbered names
        if room_name in self.room_state_manager.name_mapping:
            return self.room_state_manager.name_mapping[room_name]
        return room_name

    def _validate_rooms(self, room1: str, room2: str) -> bool:
        """Validate if both rooms exist in the room state manager"""
        if not self.room_state_manager:
            return True  # If no room state manager is provided, skip validation
        # Get standardized names
        std_room1 = self._get_standardized_room_name(room1)
        std_room2 = self._get_standardized_room_name(room2)
        rooms = self.room_state_manager.query_rooms()
        return std_room1 in rooms and std_room2 in rooms

    def _create_door(self, room1: str, room2: str) -> str:
        """Create a unique door ID for the connection between two rooms"""
        # Use standardized names for door creation
        std_room1 = self._get_standardized_room_name(room1)
        std_room2 = self._get_standardized_room_name(room2)
        return f"interior_door_{std_room1}_{std_room2}"

    def _run(self, action: str, room1: str = None, room2: str = None, has_door: bool = None):
        if action == "add":
            # Get standardized room names
            std_room1 = self._get_standardized_room_name(room1)
            std_room2 = self._get_standardized_room_name(room2)
            
            # Validate rooms before adding relationship
            if not self._validate_rooms(std_room1, std_room2):
                raise ValueError(f"One or both rooms ({std_room1}, {std_room2}) do not exist in the room state")
            # Sort room names to ensure consistent key order
            key = tuple(sorted([std_room1, std_room2]))
            self.relationships[key] = has_door
            
            # If there is a door, create door connections
            if has_door:
                door_id = self._create_door(std_room1, std_room2)
                self.door_connections[door_id] = (std_room1, std_room2)
                
        elif action == "delete":
            # Get standardized room names
            std_room1 = self._get_standardized_room_name(room1)
            std_room2 = self._get_standardized_room_name(room2)
            
            key = tuple(sorted([std_room1, std_room2]))
            if key in self.relationships:
                # If there was a door, remove door connections
                if self.relationships[key]:
                    door_id = self._create_door(std_room1, std_room2)
                    if door_id in self.door_connections:
                        del self.door_connections[door_id]
                del self.relationships[key]
                
        elif action == "update":
            # Get standardized room names
            std_room1 = self._get_standardized_room_name(room1)
            std_room2 = self._get_standardized_room_name(room2)
            
            # Validate rooms before updating relationship
            if not self._validate_rooms(std_room1, std_room2):
                raise ValueError(f"One or both rooms ({std_room1}, {std_room2}) do not exist in the room state")
            key = tuple(sorted([std_room1, std_room2]))
            if key in self.relationships:
                # If door status is changing, update door connections
                old_has_door = self.relationships[key]
                if old_has_door != has_door:
                    door_id = self._create_door(std_room1, std_room2)
                    if has_door:
                        self.door_connections[door_id] = (std_room1, std_room2)
                    elif door_id in self.door_connections:
                        del self.door_connections[door_id]
                self.relationships[key] = has_door
        return self.relationships

    def add_relationship(self, room1: str, room2: str, has_door: bool):
        return self._run("add", room1, room2, has_door)

    def delete_relationship(self, room1: str, room2: str):
        return self._run("delete", room1, room2)

    def update_relationship(self, room1: str, room2: str, has_door: bool):
        return self._run("update", room1, room2, has_door)

    def query_relationships(self):
        return self.relationships

    def query_door_connections(self):
        return self.door_connections

    def get_room_layout_info(self, room_manager: RoomStateManager) -> dict:
        """
        Generate room layout information including room IDs and connections.
        Returns a dictionary with:
        - room_ids: list of room type IDs (e.g., [1, 3, 3, ...])
        - connections: list of room index pairs that are connected (e.g., [[0, 1], ...])
        - door_connections: list of door connections (e.g., [[room1_idx, door_idx, room2_idx], ...])
        """
        # Get all unique rooms from relationships
        all_rooms = set()
        for (room1, room2) in self.relationships.keys():
            all_rooms.add(room1)
            all_rooms.add(room2)
        
        # Convert room names to room IDs
        room_ids = []
        room_name_to_index = {}  # Map room names to their indices in room_ids list
        
        for room_name in sorted(all_rooms):
            room_info = room_manager.get_room_by_name(room_name)
            if room_info:
                room_type = room_info["type"]
                room_type_id = room_id.get(room_type, 0)  # Get ID from room_id mapping
                room_ids.append(room_type_id)
                room_name_to_index[room_name] = len(room_ids) - 1
        
        # Generate connections list
        connections = []
        
        for (room1, room2), has_door in self.relationships.items():
            if has_door:  # Only include connections with doors
                idx1 = room_name_to_index.get(room1)
                idx2 = room_name_to_index.get(room2)
                if idx1 is not None and idx2 is not None:
                    connections.append([idx1, 1, idx2])
                    
                    # Add door connection
                    door_id = self._create_door(room1, room2)
                    door_idx = len(room_ids)  # Door index is after all room indices
                    room_ids.append(room_id["interior_door"])  # Add door to room_ids
                    connections.append([idx1, 1, door_idx])
                    connections.append([door_idx, 1, idx2])
        
        return {
            "room_ids": room_ids,
            "connections": connections,
        }

class RoomGraphVisualizer:
    def __init__(self, relationship_manager: RoomRelationshipManager):
        self.relationship_manager = relationship_manager

    def visualize_relationships(self):
        # Create a new graph
        G = nx.Graph()
        
        # Add nodes and edges (only for rooms connected by doors)
        for (room1, room2), has_door in self.relationship_manager.query_relationships().items():
            if has_door:  # Only add edges for rooms connected by doors
                G.add_node(room1)
                G.add_node(room2)
                G.add_edge(room1, room2)
        
        # Create the plot with a white background
        plt.figure(figsize=(12, 8))
        plt.gca().set_facecolor('white')
        plt.gcf().set_facecolor('white')
        
        # Use a more structured layout
        pos = nx.spring_layout(G, k=4.0, iterations=100, scale=4.0)  # Increase k value to make node spacing larger, increase iterations to make layout more stable, increase scale to make overall layout more dispersed
        
        # Draw nodes with better styling
        nx.draw_networkx_nodes(G, pos,
                            node_color='#f0f9ff',  # Light blue background
                            node_size=7000,
                            alpha=1.0,
                            linewidths=2,
                            edgecolors='#3498db')  # Dark blue border
        
        # Draw edges
        nx.draw_networkx_edges(G, pos,
                            edge_color='#2ecc71',  # Use soft green color
                            width=2.5)
        
        # Add labels with better font
        nx.draw_networkx_labels(G, pos,
                            font_size=12,
                            font_family='sans-serif',
                            font_weight='bold')
        
        # Remove axes
        plt.axis('off')
        
        # Add some padding around the graph
        plt.margins(0.2)
        
        # Save the plot to a PIL Image
        fig = plt.gcf()
        canvas = fig.canvas
        canvas.draw()
        
        # Get the RGBA buffer from the figure
        width, height = fig.get_size_inches() * fig.get_dpi()
        width = int(width)
        height = int(height)
        
        # Convert canvas to image
        image_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        image_array = image_array.reshape((height, width, 4))
        
        # Convert RGBA to RGB
        rgb_array = image_array[:, :, :3]
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(rgb_array)
        
        plt.close()
        return image

class RoomRelationshipLLM:
    def __init__(self, relationship_manager: RoomRelationshipManager, llm):
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

# Instantiate relationship manager
relationship_tool = RoomRelationshipManager(room_tool)
relationship_llm = RoomRelationshipLLM(relationship_tool, model)

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

# Use custom Memory class
memory = CustomSummaryMemory(
    system=system_message,
    llm=model,
    max_tokens=2000,
    room_state_llm=room_state_llm,  # Pass room_state_llm instance
    relationship_llm=relationship_llm  # Pass relationship_llm instance
)

# message = 'what\'s your name?'
# memory.add_user_message(message)
# response = model.invoke(memory.buffer)
# memory.add_ai_message(response.content)
# room_state_llm.process_conversation(f"User: {message}\nAI: {response.content}")
# message = 'what\'s your name?'
# memory.add_user_message(message)
# response = model.invoke(memory.buffer)
# memory.add_ai_message(response.content)

# Add phase tracker
class DesignPhaseTracker:
    def __init__(self):
        self.current_phase = 1  # Initially phase 1
        self.phase_names = {
            1: "Room Confirmation Phase",
            2: "Room Relationship Confirmation Phase",
            3: "Room Layout Diagram Generation Phase"
        }
    
    def get_current_phase(self) -> int:
        return self.current_phase
    
    def get_phase_name(self) -> str:
        return self.phase_names.get(self.current_phase, "Unknown Phase")
    
    def move_to_next_phase(self) -> bool:
        if self.current_phase < 3:
            self.current_phase += 1
            return True
        return False
    
    def is_phase_3(self) -> bool:
        return self.current_phase == 3

# Instantiate phase tracker
phase_tracker = DesignPhaseTracker()

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

# Instantiate phase transition LLM
phase_transition_llm = PhaseTransitionLLM(model, room_state_llm, relationship_llm)

def process(message, history):
    # Add user message to memory
    message = message.get("text", "")
    memory.add_user_message(message)
    response = model.invoke(memory.buffer)
    memory.add_ai_message(response.content)
    
    # First check if phase transition is needed
    transition_decision = phase_transition_llm.should_transition(
        phase_tracker.get_current_phase(),
        f"User: {message}\nAI: {response.content}"
    )
    
    # If decided to transition to next phase, update phase first
    if transition_decision["should_transition"]:
        phase_tracker.move_to_next_phase()
        # If there are missing key elements, add them to response
        if transition_decision["missing_elements"]:
            response.content += "\n\nNote: " + ", ".join(transition_decision["missing_elements"])
    
    # Get current phase (may be updated)
    current_phase = phase_tracker.get_current_phase()
    
    # Execute corresponding state updates based on current phase
    if current_phase == 1:
        # Phase 1: Update room state
        room_state_llm.process_conversation(f"User: {message}\nAI: {response.content}")
        visualization = None
    
    elif current_phase == 2:
        # Phase 2: Update room relationships
        relationship_results, visualization = relationship_llm.process_conversation(f"User: {message}\nAI: {response.content}")
    
    elif current_phase == 3:
        # Phase 3: Generate layout information
        layout_info = relationship_tool.get_room_layout_info(room_tool)
        # In phase 3, we only return layout information without additional descriptions
        response.content = "Generating layout diagram..."
        visualization = None
    
    # Build return information
    result = []
    
    # Add text message
    result.append(response.content)
    
    # Add phase information
    result.append(f"Current Phase: {phase_tracker.get_phase_name()}")
    
    # If in phase 3, add layout information
    if current_phase == 3:
        result.append(f"Layout Info: {layout_info}")
    
    # If in phase 2, add visualization (if available)
    if current_phase == 2 and visualization is not None:
        result.append(gr.Image(visualization))
    
    return result

demo = gr.ChatInterface(
    title="Floor Plan Designer",
    fn=process, 
    multimodal=True,
    type="messages",
    description="This is a floor plan designer. You can use it to design your floor plan. You can also use it to ask questions about the floor plan.",
)


demo.launch()