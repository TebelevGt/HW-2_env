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

    # Format the matrix as a string.
    # threshold=np.inf ensures the full matrix is printed even if it's large.
    matrix_str = np.array2string(adjacency_matrix, separator=", ", threshold=np.inf)

    return (
        "You are an expert in graph algorithms. Your task is to find the shortest path in a weighted undirected graph.\n\n"
        "### Rules and Definitions\n"
        "1. **Graph Representation**:\n"
        f"   - The graph consists of {n_nodes} nodes, numbered from 0 to {n_nodes - 1}.\n"
        "   - The graph is provided as an adjacency matrix.\n"
        "   - The matrix is symmetric (undirected graph).\n"
        "   - A value of `0` at position (i, j) indicates **no direct edge** between node i and node j.\n"
        "   - A positive integer at position (i, j) represents the **weight** (cost) of the edge between node i and node j.\n"
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
        f"Adjacency Matrix:\n{matrix_str}\n"
    )
