import networkx as nx
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
import numpy as np

def max_connections(graph, module):
    """Return the maximum number of connections for a given module type."""
    module_index = module['index']
    if module_index == 0:
        return 5  # Core module
    elif graph.nodes[module_index]["type"] == "BRICK":
        return 5
    elif graph.nodes[module_index]["type"] == "HINGE":
        # For simplicity, assume all other modules are bricks with max 4 connections
        return 2
    
def num_connections(graph, module):
    """Return the current number of connections for a given module."""
    module_index = module['index']
    print("Module Index", module_index)
    return len(list(graph.neighbors(module_index)))
    
def available_faces(graph, module):
    """Return the available faces for a given module."""
    module_index = module['index']
    if graph.nodes[module_index]["type"] == "BRICK":
        all_faces = {"FRONT", "RIGHT", "LEFT", "TOP"}
    elif graph.nodes[module_index]["type"] == "CORE":   
        all_faces = {"FRONT", "BACK", "RIGHT", "LEFT", "TOP"}
    else:  # HINGE or other types
        all_faces = {"FRONT"}
    connected_faces = set()
    
    for neighbor in graph.neighbors(module_index):
        edge_data = graph.get_edge_data(module_index, neighbor)
        if edge_data and "face" in edge_data:
            connected_faces.add(edge_data["face"])
        
    available_faces = list(all_faces - connected_faces)
    print("Available faces for module", module_index, ":", available_faces)
    
    return available_faces

def process_node(graph, current_node, index, nodes_to_process):
    """Process the current node to add new modules and return a list of nodes to process."""
    faces = available_faces(graph, current_node)

    if available_faces == 0:
        return graph, [], index, nodes_to_process  # No more connections can be added
    else:      
        chance_brick = 0.35
        chance_hinge = 0.5
        for face in faces:
            if np.random.rand() < chance_brick:
                graph.add_nodes_from([(index, {"type": "BRICK", "rotation": "DEG_0"})])
                graph.add_edges_from([(current_node['index'], index, {"face": face})])
                nodes_to_process.append({'index': index, 'level': current_node['level'] + 1})
                index += 1
            elif np.random.rand() < chance_hinge:
                rotation_choices = ["DEG_90", "DEG_180"]
                rotation = np.random.choice(rotation_choices)
                graph.add_nodes_from([(index, {"type": "HINGE", "rotation": rotation})])
                graph.add_edges_from([(current_node['index'], index, {"face": face})])
                nodes_to_process.append({'index': index, 'level': current_node['level'] + 1})
                index += 1
            else:
                graph.add_nodes_from([(index, {"type": "NONE", "rotation": "DEG_0"})])
                graph.add_edges_from([(current_node['index'], index, {"face": face})])
                index += 1

    return graph, nodes_to_process, index, nodes_to_process

# Create a new robot body with random connections
def new_robot():
    """Create a new robot body."""
    graph = nx.Graph()
    # Add the core node
    graph.add_nodes_from([(0, {"type": "CORE", "rotation": "DEG_0"})])

    max_levels = 3  # Maximum levels of hierarchy
    index = 1 # Start adding modules from index 1

    nodes_to_process = [{'index': 0, 'level': 0}]  # Start with the core node
    current_node = nodes_to_process.pop(0)

    while num_connections(graph, current_node) < max_connections(graph, current_node) or nodes_to_process:

        print("Number of connections for node", current_node, ":", num_connections(graph, current_node))
        
        if num_connections(graph, current_node) < max_connections(graph, current_node):
            if current_node['level'] < max_levels:
                graph, nodes_to_process, index, nodes_to_process = process_node(graph, current_node, index, nodes_to_process)
            elif current_node['level'] >= max_levels and nodes_to_process:
                print("Max levels reached for node", current_node)
                current_node = nodes_to_process.pop(0)

            print("Nodes to process:", nodes_to_process)
        
        if nodes_to_process:
            current_node = nodes_to_process.pop(0)
        else:
            break  # No more nodes to process
        
    indexes = range(index-1)
    print("Indexes:", list(indexes))
    print("Edges:", graph.edges(data=True))

    return construct_mjspec_from_graph(graph)