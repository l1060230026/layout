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