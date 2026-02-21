import re
import random
import networkx as nx
from typing import Optional, List
from base_classes import DataHolder, Env, Verifier


class PathVerifier(Verifier):
    def extract_answer(self, test_solution: str) -> str:
        """Достаем ответ из тегов <answer>...</answer>"""
        match = re.search(r"<answer>(.*?)</answer>", test_solution, re.DOTALL)
        return match.group(1).strip() if match else ""

    def verify(self, data: DataHolder, test_solution: str) -> bool:
        """Проверяем, что путь существует и его стоимость равна минимальной"""
        pred_str = self.extract_answer(test_solution)
        if not pred_str:
            return False

        try:
            # Парсим путь из строки "0, 1, 2" в список [0, 1, 2]
            pred_path = [int(x.strip()) for x in pred_str.split(",")]

            # Восстанавливаем граф из метаданных для проверки
            graph = nx.from_numpy_array(data.metadata["matrix"])

            # Считаем стоимость предложенного пути
            cost = 0
            for i in range(len(pred_path) - 1):
                u, v = pred_path[i], pred_path[i + 1]
                if not graph.has_edge(u, v):
                    return False  # LLM выдумала несуществующее ребро
                cost += graph[u][v]["weight"]

            # Проверяем, что начало/конец верные, и стоимость оптимальная
            is_valid_start_end = pred_path[0] == data.metadata["start"] and pred_path[-1] == data.metadata["end"]
            is_optimal = cost == data.metadata["optimal_cost"]

            return is_valid_start_end and is_optimal
        except Exception:
            return False  # Если LLM выдала мусор вместо чисел


class PathEnv(Env):
    def __init__(self):
        super().__init__(name="ShortestPath", verifier=PathVerifier)

    def extract_answer(self, test_solution: str) -> str:
        return self.verifier.extract_answer(test_solution)

    def generate(
        self, num_of_questions: int = 100, max_attempts: int = 100, difficulty: Optional[int] = 1
    ) -> List[DataHolder]:
        data_list = []
        # Сложность (1-10) влияет на количество вершин: diff 1 -> 5 вершин, diff 10 -> 23 вершины
        n_nodes = (difficulty or 1) * 2 + 3

        for _ in range(num_of_questions):
            for _ in range(max_attempts):
                # Генерируем случайный граф (p=0.4 - плотность связей)
                G = nx.fast_gnp_random_graph(n_nodes, p=0.4)

                # Нам нужен только связный граф, чтобы путь гарантированно существовал
                if nx.is_connected(G):
                    # Навешиваем случайные веса от 1 до 10 на ребра
                    for u, v in G.edges():
                        G.edges[u, v]["weight"] = random.randint(1, 10)

                    # Выбираем случайные старт и конец
                    start, end = random.sample(list(G.nodes()), 2)

                    # Делегируем поиск решения networkx (Дейкстра под капотом)
                    optimal_path = nx.shortest_path(G, source=start, target=end, weight="weight")
                    optimal_cost = nx.shortest_path_length(G, source=start, target=end, weight="weight")

                    # Формируем матрицу и промпт
                    mat = nx.to_numpy_array(G, nonedge=0).astype(int)
                    prompt = (
                        f"You are given a weighted undirected graph represented by an adjacency matrix.\n"
                        f"Nodes are numbered from 0 to {n_nodes - 1}. A value of 0 means no direct edge.\n"
                        f"Matrix:\n{mat}\n\n"
                        f"Find the shortest path from node {start} to node {end}.\n"
                        f"Provide your final answer as a comma-separated list of node indices inside the <answer> tags."
                    )

                    # Эталонный ответ (просто для логов/сравнения)
                    ans_str = ", ".join(map(str, optimal_path))

                    data_list.append(
                        DataHolder(
                            question=prompt,
                            answer=ans_str,
                            difficulty=difficulty,
                            metadata={"matrix": mat, "start": start, "end": end, "optimal_cost": optimal_cost},
                        )
                    )
                    break  # Граф успешно сгенерирован, идем к следующему вопросу

        return data_list
