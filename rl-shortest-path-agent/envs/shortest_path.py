import re
import random
import networkx as nx
from typing import Optional, List
from envs.base_classes import Data, Env, Verifier
from envs.prompts import generate_shortest_path_prompt


class PathVerifier(Verifier):
    def extract_answer(self, test_solution: str) -> str:
        """Достаем ответ из тегов <answer>...</answer>"""
        match = re.search(r"<answer>(.*?)</answer>", test_solution, re.DOTALL)
        return match.group(1).strip() if match else ""

    def verify(self, data: Data, test_solution: str) -> bool:
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
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = 1,
        n_nodes: Optional[int] = None,
        edge_prob: float = 0.4,
        min_weight: int = 1,
        max_weight: int = 10,
    ) -> List[Data]:
        data_list = []

        # Если n_nodes не задан явно, вычисляем его на основе difficulty
        if n_nodes is None:
            n_nodes = (difficulty or 1) * 2 + 3

        for _ in range(num_of_questions):
            for _ in range(max_attempts):
                # Генерируем случайный граф
                G = nx.fast_gnp_random_graph(n_nodes, p=edge_prob)

                # Нам нужен только связный граф, чтобы путь гарантированно существовал
                if nx.is_connected(G):
                    # Навешиваем случайные веса на ребра
                    for u, v in G.edges():
                        G.edges[u, v]["weight"] = random.randint(min_weight, max_weight)

                    # Выбираем случайные старт и конец
                    start, end = random.sample(list(G.nodes()), 2)

                    # Делегируем поиск решения networkx (Дейкстра под капотом)
                    optimal_path = nx.shortest_path(G, source=start, target=end, weight="weight")
                    optimal_cost = nx.shortest_path_length(G, source=start, target=end, weight="weight")

                    # Формируем матрицу и промпт
                    mat = nx.to_numpy_array(G, nonedge=0).astype(int)
                    prompt = generate_shortest_path_prompt(mat, start, end)

                    # Эталонный ответ (просто для логов/сравнения)
                    ans_str = ", ".join(map(str, optimal_path))

                    data_list.append(
                        Data(
                            question=prompt,
                            answer=ans_str,
                            difficulty=difficulty,
                            metadata={"matrix": mat, "start": start, "end": end, "optimal_cost": optimal_cost},
                        )
                    )
                    break  # Граф успешно сгенерирован, идем к следующему вопросу

        return data_list

    def visualize(self, data: Data, save_path: Optional[str] = None):
        """
        Визуализирует граф задачи.
        :param data: объект Data с метаданными графа
        :param save_path: путь для сохранения изображения (опционально)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Для визуализации требуется библиотека matplotlib (pip install matplotlib).")
            return

        # Восстанавливаем граф из матрицы смежности
        G = nx.from_numpy_array(data.metadata["matrix"])
        start = data.metadata["start"]
        end = data.metadata["end"]

        pos = nx.spring_layout(G, seed=42)  # seed для стабильного отображения

        plt.figure(figsize=(8, 6))

        # Рисуем узлы, ребра и метки
        nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500)
        nx.draw_networkx_nodes(G, pos, nodelist=[start], node_color="green", label="Start")
        nx.draw_networkx_nodes(G, pos, nodelist=[end], node_color="red", label="End")
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "weight"))

        plt.legend()
        plt.title(f"Shortest Path Task: {start} -> {end}")

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
