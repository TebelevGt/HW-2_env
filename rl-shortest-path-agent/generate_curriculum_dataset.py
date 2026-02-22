import os
from envs.shortest_path import PathEnv, ShortestPathDataset


def main():
    env = PathEnv()

    # Настройка Curriculum Learning
    # Мы делим датасет на 3 части с разным уровнем сложности.
    # difficulty=1 -> ~4 узла, difficulty=10 -> ~18 узлов.
    stages = [
        # 1. Разминка: 20% простых задач (сложность 1-3)
        {"difficulty": (1, 3), "ratio": 0.2},
        # 2. Основная часть: 50% средних задач (сложность 4-7)
        {"difficulty": (4, 7), "ratio": 0.5},
        # 3. Челлендж: 30% сложных задач (сложность 8-12)
        {"difficulty": (8, 12), "ratio": 0.3},
    ]

    total_samples = 2000
    output_path = "data/train_curriculum.pkl"

    print(f"Generating Curriculum Dataset ({total_samples} samples)...")
    dataset = ShortestPathDataset.create_curriculum(env, total_samples, stages)

    dataset.save(output_path)
    print(f"Done! Dataset saved to {output_path}")


if __name__ == "__main__":
    main()
