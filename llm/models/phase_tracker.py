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