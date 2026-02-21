import numpy as np


def generate_shortest_path_prompt(adjacency_matrix: np.ndarray, start_node: int, end_node: int) -> str:
    """
    Generates a comprehensive, LLM-friendly prompt for the Shortest Path problem,
    forcing Chain-of-Thought reasoning.
    """
    n_nodes = adjacency_matrix.shape[0]

    # Более человекочитаемый формат: "Node 0 is connected to: Node 1 (weight 5), Node 2 (weight 3)"
    edge_lines = []
    for i in range(n_nodes):
        neighbors = []
        for j in range(n_nodes):
            weight = adjacency_matrix[i, j]
            if weight > 0:
                # Обязательно приводим к int, чтобы не было float-артефактов типа 5.0
                neighbors.append(f"Node {j} (weight {int(weight)})")

        if neighbors:
            edge_lines.append(f"Node {i} is connected to: " + ", ".join(neighbors))
        else:
            edge_lines.append(f"Node {i} is isolated.")

    graph_str = "\n".join(edge_lines)

    # Формируем промпт с четкой инструкцией и примером CoT
    prompt = (
        f"You are an expert at graph pathfinding. Find the shortest path from Node {start_node} to Node {end_node} in the weighted undirected graph below.\n\n"
        "Graph connections:\n"
        f"{graph_str}\n\n"
        "Instructions:\n"
        "1. You MUST think step-by-step inside <reasoning> tags. Explore neighbors, track the cumulative costs from the start node, and keep track of the current shortest paths.\n"
        "2. After your reasoning, output ONLY the final sequence of node indices for the shortest path, comma-separated, inside <answer> tags.\n\n"
        "Example output format:\n"
        "<reasoning>\n"
        "Starting at Node A.\n"
        "Neighbors of Node A are Node B (weight 2) and Node C (weight 5).\n"
        "Current paths:\n"
        "- Path A->B: cost 2\n"
        "- Path A->C: cost 5\n"
        "Exploring from Node B (lowest current cost 2). Neighbors of Node B are Node D (weight 3). New path A->B->D: cost 2+3=5.\n"
        "...and so on, tracking the lowest cumulative cost until reaching the target node.\n"
        "</reasoning>\n"
        "<answer>A, B, D</answer>"
    )

    return prompt
