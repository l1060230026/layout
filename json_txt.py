import os
import json
from openai import OpenAI


client = OpenAI(api_key="", base_url="")

def prompt_chat(prompt):
    # Call ChatGPT chat API
    response = client.chat.completions.create(
    model="gpt-4o-mini",  # or gpt-4
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "balcony": 5, "entrance": 6, "dining room": 7, "study room": 8,
              "storage": 10 , "unknown": 16,}

# Create reverse mapping from number to room type
ROOM_CLASS_REVERSE = {v: k for k, v in ROOM_CLASS.items()}

def get_room_name(room_type, index, total_count):
    """Get room name with index for rooms of the same type"""
    room_type_name = ROOM_CLASS_REVERSE.get(room_type, "unknown")
    if total_count > 1:
        return f"{room_type_name}_{index}"
    return room_type_name

def describe_floor_plan(room_types, connections):
    # Count occurrences of each room type
    room_type_counts = {}
    room_names = []
    
    # First pass: count total occurrences of each room type
    for room_type in room_types:
        if room_type not in room_type_counts:
            room_type_counts[room_type] = 1
        else:
            room_type_counts[room_type] += 1
    
    # Second pass: create room names
    current_counts = {}
    for room_type in room_types:
        if room_type not in current_counts:
            current_counts[room_type] = 1
        else:
            current_counts[room_type] += 1
        
        room_names.append(get_room_name(room_type, current_counts[room_type], room_type_counts[room_type]))
    
    room_names_str = ", ".join(room_names)

    
    connections_str = ""
    # Print connections
    for connection in connections:
        room1, room2 = connection
        connections_str += f"{room_names[room1]} connects to {room_names[room2]}\n"
        
    prompt = f"Please describe the room design in natural language based on room types and connections.\nRoom types: {room_names_str}\nConnections: {connections_str}"
    description = prompt_chat(prompt)
    return description

file_names = os.listdir('datasets/rplan')[:100]

os.makedirs('datasets/rplan_txt', exist_ok=True)

for file_name in file_names:
    with open(f'datasets/rplan/{file_name}', 'r') as f:
        data = json.load(f)
        

    room_type = data['room_types']

    room_end = room_type.index(17)
    room_type = room_type[:room_end]

    graph = data['graph']
    graph = [item for item in graph if item[0] < room_end and item[1] < room_end]

    # Remove duplicate connections by sorting each pair and using a set
    graph = list(set(tuple(sorted(pair)) for pair in graph))
    graph = [list(pair) for pair in graph]

    description = describe_floor_plan(room_type, graph)
    
    data['description'] = description

    with open(f'datasets/rplan_txt/{file_name}', 'w') as f:
        json.dump(data, f)



