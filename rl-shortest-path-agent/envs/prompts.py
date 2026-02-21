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

    # Compact Adjacency List format: "0: 1(5) 2(3)" to save tokens
    edge_lines = []
    for i in range(n_nodes):
        neighbors = []
        for j in range(n_nodes):
            weight = adjacency_matrix[i, j]
            if weight > 0:
                neighbors.append(f"{j}({weight})")
        if neighbors:
            edge_lines.append(f"{i}: " + " ".join(neighbors))
        else:
            edge_lines.append(f"{i}: -")

    graph_str = "\n".join(edge_lines)

    return (
        f"Find the shortest path from Node {start_node} to Node {end_node} in the weighted undirected graph below.\n"
        "The graph is represented as an adjacency list where 'i: j(w)' means there is an edge between node i and node j with weight w.\n\n"
        f"Graph:\n{graph_str}\n\n"
        "Output the sequence of node indices for the shortest path, comma-separated, inside <answer> tags.\n"
        "Example: <answer>0, 2, 5, 3</answer>"
    )
