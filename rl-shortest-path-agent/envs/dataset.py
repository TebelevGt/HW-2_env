import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional

from envs.base_classes import Data
from envs.shortest_path import PathEnv


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
    train_ds = ShortestPathDataset.create(env, num_samples=5000, n_nodes=15, edge_prob=0.3)
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


# if __name__ == "__main__":
# create_benchmark_datasets()
