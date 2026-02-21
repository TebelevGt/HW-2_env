import numpy as np


def generate_shortest_path_prompt(adjacency_matrix: np.ndarray, start_node: int, end_node: int) -> str:
    """
    Generates a comprehensive prompt for the Shortest Path problem, including rules and input data.

    Args:
        adjacency_matrix (np.ndarray): The weighted adjacency matrix of the graph.
        start_node (int): The starting node index.
        end_node (int): The destination node index.

    Returns:
        str: A formatted string containing the system prompt, rules, and problem instance.
    """
    n_nodes = adjacency_matrix.shape[0]

    # Convert matrix to Adjacency List format to save tokens
    edge_lines = []
    for i in range(n_nodes):
        neighbors = []
        for j in range(n_nodes):
            if adjacency_matrix[i, j] > 0:
                neighbors.append(f"Node {j} (weight {adjacency_matrix[i, j]})")
        if neighbors:
            edge_lines.append(f"Node {i}: " + ", ".join(neighbors))
        else:
            edge_lines.append(f"Node {i}: No connections")

    graph_str = "\n".join(edge_lines)

    return (
        "You are an expert in graph algorithms. Your task is to find the shortest path in a weighted undirected graph.\n\n"
        "### Rules and Definitions\n"
        "1. **Graph Representation**:\n"
        f"   - The graph consists of {n_nodes} nodes, numbered from 0 to {n_nodes - 1}.\n"
        "   - The graph is provided as an adjacency list.\n"
        "   - For each node, its neighbors and the edge weights are listed.\n"
        "   - The graph is undirected (if A is connected to B, B is connected to A with the same weight).\n"
        "2. **Objective**:\n"
        f"   - Find the path from **Node {start_node}** to **Node {end_node}** with the minimum total weight.\n"
        "   - The total weight is the sum of the weights of the edges in the path.\n"
        "3. **Output Format**:\n"
        "   - Provide your final answer strictly as a comma-separated list of node indices.\n"
        "   - The list must be enclosed within `<answer>` and `</answer>` tags.\n"
        "   - Example: <answer>0, 2, 5, 3</answer>\n\n"
        "### Problem Instance\n"
        f"Start Node: {start_node}\n"
        f"End Node: {end_node}\n"
        f"Graph Description:\n{graph_str}\n"
    )
