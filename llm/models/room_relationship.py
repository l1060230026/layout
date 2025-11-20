from utils.constants import room_id

class RoomRelationshipManager:
    def __init__(self, room_state_manager=None):
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

    def get_room_layout_info(self, room_manager) -> dict:
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