import re
import os
import random
import numpy as np
import networkx as nx
from typing import Optional, List
from envs.base_classes import Data, Env, Verifier
from envs.prompts import generate_path_prompt, SYSTEM_PROMPT
import pickle
from torch.utils.data import Dataset


class PathVerifier(Verifier):
    def extract_answer(self, test_solution: str) -> str:
        """Достаем ответ из тегов <answer>...</answer>"""
        match = re.search(r"<answer>(.*?)</answer>", test_solution, re.DOTALL)
        return match.group(1).strip() if match else ""

    def verify(self, data: Data, test_solution: str) -> bool:
        """Проверяем, что путь существует и ведет от старта к финишу"""
        pred_str = self.extract_answer(test_solution)
        if not pred_str:
            return False

        try:
            # Парсим путь из строки "0, 1, 2" в список [0, 1, 2]
            pred_path = [int(x.strip()) for x in pred_str.split(",")]

            # Восстанавливаем граф из метаданных для проверки
            graph = nx.from_numpy_array(data.metadata["matrix"])

            # Проверяем существование ребер
            for i in range(len(pred_path) - 1):
                u, v = pred_path[i], pred_path[i + 1]
                if not graph.has_edge(u, v):
                    return False  # LLM выдумала несуществующее ребро

            # Проверяем, что начало/конец верные
            is_valid_start_end = pred_path[0] == data.metadata["start"] and pred_path[-1] == data.metadata["end"]

            return is_valid_start_end
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
            n_nodes = int((difficulty or 1) * 1.5 + 3)

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
                    prompt = generate_path_prompt(mat, start, end)

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


class ShortestPathDataset(Dataset):
    """
    PyTorch Dataset для хранения и загрузки задач Shortest Path.
    Гарантирует воспроизводимость тестов, так как данные генерируются один раз и сохраняются.
    """

    def __init__(self, data: List[Data]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Data:
        # Возвращаем сам объект Data.
        # Внимание: стандартный DataLoader попытается превратить это в тензор и упадет.
        # Используйте collate_data_fn.
        return self.data[idx]

    def save(self, filepath: str):
        """Сохраняет датасет в файл (pickle)."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.data, f)
        print(f"Dataset saved: {filepath} ({len(self.data)} samples)")

    @classmethod
    def load(cls, filepath: str) -> "ShortestPathDataset":
        """Загружает датасет из файла."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return cls(data)

    @classmethod
    def create(cls, env: PathEnv, num_samples: int, **kwargs) -> "ShortestPathDataset":
        """
        Создает новый датасет, генерируя данные через среду.
        kwargs передаются в env.generate() (например, difficulty, n_nodes, edge_prob).
        """
        print(f"Generating {num_samples} samples with params: {kwargs}...")
        raw_data = env.generate(num_of_questions=num_samples, **kwargs)
        return cls(raw_data)


def collate_data_fn(batch: List[Data]) -> List[Data]:
    """
    Custom collate_fn для DataLoader.
    Предотвращает попытки PyTorch превратить объекты Data в тензоры.
    Возвращает просто список объектов.
    """
    return batch


def create_benchmark_datasets(output_dir: str = "data"):
    """
    Генерирует стандартный набор датасетов для обучения и тестирования.
    Запускайте эту функцию один раз, чтобы создать файлы.
    """
    # Фиксируем сид для полной воспроизводимости генерации самих датасетов
    random.seed(42)
    np.random.seed(42)

    env = PathEnv()

    # 1. Train Dataset (Разнообразный, посложнее)
    # Генерируем 5000 задач с ~15 узлами
    train_ds = ShortestPathDataset.create(env, num_samples=5000, n_nodes=12, edge_prob=0.3)
    train_ds.save(os.path.join(output_dir, "train_v1.pkl"))

    # 2. Test Datasets (Разной сложности для оценки)

    # Easy: Маленькие графы (difficulty=1 -> ~5 узлов)
    test_easy = ShortestPathDataset.create(env, num_samples=200, difficulty=1)
    test_easy.save(os.path.join(output_dir, "test_easy.pkl"))

    # Medium: Средние графы (difficulty=5 -> ~13 узлов)
    test_medium = ShortestPathDataset.create(env, num_samples=200, difficulty=5)
    test_medium.save(os.path.join(output_dir, "test_medium.pkl"))

    # Hard: Большие графы (difficulty=10 -> ~23 узла)
    test_hard = ShortestPathDataset.create(env, num_samples=200, difficulty=10)
    test_hard.save(os.path.join(output_dir, "test_hard.pkl"))


def get_shortest_path_dataset(filepath: str):
    """Загружает сохраненный ShortestPathDataset и конвертирует его для GRPOTrainer"""
    import datasets

    custom_dataset = ShortestPathDataset.load(filepath)

    data_dict = {
        "prompt": [],
        "answer": [],
        "matrix": [],
        "start": [],
        "end": [],
        "optimal_cost": [],
    }

    for item in custom_dataset.data:
        chat_prompt = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": item.question}]
        data_dict["prompt"].append(chat_prompt)
        data_dict["answer"].append(item.answer)

        # Сохраняем метаданные для correctness_reward_func
        data_dict["matrix"].append(item.metadata["matrix"].tolist())
        data_dict["start"].append(item.metadata["start"])
        data_dict["end"].append(item.metadata["end"])
        data_dict["optimal_cost"].append(item.metadata["optimal_cost"])

    return datasets.Dataset.from_dict(data_dict)
