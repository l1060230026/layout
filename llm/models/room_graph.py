import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class RoomGraphVisualizer:
    def __init__(self, relationship_manager):
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