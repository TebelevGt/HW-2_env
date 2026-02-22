import pickle
import re
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from typing import Union, Dict, Any
from envs.shortest_path import ShortestPathDataset
from envs.prompts import SYSTEM_PROMPT

import torch


# 1. Создаем обертку, которая принимает текст, генерирует ответ и возвращает текст
class ModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, prompt: str) -> str:
        # Токенизируем промпт
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Генерируем токены
        outputs = self.model.generate(**inputs, max_new_tokens=512, temperature=0.0)

        # Обрезаем сам промпт из ответа (чтобы осталась только генерация)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]

        # Декодируем в чистую строку
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_xml_tag(text: str, tag: str) -> str:
    """Извлекает содержимое тегов <tag>...</tag>."""
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def evaluate_agent(
    model_input: Union[str, Any],
    dataset_input: Union[str, ShortestPathDataset],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    generate_kwargs: Dict[str, Any] = None,
) -> Dict[str, float]:
    """
    Оценивает модель (загруженную из pickle или переданную как объект) на датасете ShortestPathDataset.

    Args:
        model_input: Путь к pickle-файлу с моделью или сам объект модели.
                     Ожидается, что объект вызываемый (prompt -> str)
                     или имеет метод `generate` / `predict`.
        dataset_input: Путь к pickle-файлу датасета или объект ShortestPathDataset.
        device: Устройство для загрузки (если применимо).
        generate_kwargs: Дополнительные аргументы генерации (например, max_new_tokens).

    Returns:
        Словарь с метриками:
        - accuracy: Доля валидных путей, достигших цели.
        - format_compliance: Доля ответов с корректными тегами <answer>.
        - optimal_rate: Доля путей, совпадающих по стоимости с оптимальным.
        - avg_optimality_gap: Средняя разница между стоимостью предсказанного и оптимального пути.
        - avg_reasoning_len: Средняя длина рассуждений (в символах).
    """

    # 1. Загрузка датасета
    if isinstance(dataset_input, str):
        print(f"Loading dataset from {dataset_input}...")
        dataset = ShortestPathDataset.load(dataset_input)
    else:
        dataset = dataset_input

    # 2. Загрузка модели
    if isinstance(model_input, str):
        print(f"Loading model from {model_input}...")
        try:
            # Пробуем загрузить как обычный pickle
            model = load_pickle(model_input)
        except Exception as e:
            print(f"Pickle load failed ({e}), trying torch.load...")
            model = torch.load(model_input, map_location=device)
    else:
        model = model_input

    # Хелпер для инференса
    if generate_kwargs is None:
        generate_kwargs = {"max_new_tokens": 512, "temperature": 0.0}

    def run_inference(prompt: str) -> str:
        # Адаптер для разных типов моделей
        try:
            if callable(model):
                return model(prompt)
            elif hasattr(model, "generate_content"):  # Google GenAI style
                return model.generate_content(prompt).text
            elif hasattr(model, "predict"):  # Sklearn / Custom style
                return model.predict(prompt)
            elif hasattr(model, "generate"):  # HuggingFace style (simplified wrapper)
                # Если это пайплайн или обертка
                out = model.generate(prompt, **generate_kwargs)
                return str(out)
            else:
                return str(model)
        except Exception as e:
            print(f"Inference error: {e}")
            return ""

    # 3. Цикл оценки
    metrics = {
        "total": 0,
        "format_error": 0,
        "valid_path": 0,
        "optimal_count": 0,
        "avg_optimality_gap": [],
        "avg_reasoning_len": [],
    }

    print(f"Starting evaluation on {len(dataset)} samples...")

    for data in tqdm(dataset):
        metrics["total"] += 1

        # Формируем промпт
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {data.question}\n\nAssistant:"

        # Генерация
        response = run_inference(prompt)

        # Извлечение ответа
        reasoning = extract_xml_tag(response, "reasoning")
        answer = extract_xml_tag(response, "answer")

        metrics["avg_reasoning_len"].append(len(reasoning))

        if not answer:
            metrics["format_error"] += 1
            continue

        # Парсинг пути
        try:
            clean_ans = re.sub(r"[^\d,]", "", answer)
            pred_path = [int(x.strip()) for x in clean_ans.split(",") if x.strip()]
        except ValueError:
            metrics["format_error"] += 1
            continue

        if not pred_path:
            metrics["format_error"] += 1
            continue

        # Проверка пути
        matrix = data.metadata["matrix"]
        start_node = data.metadata["start"]
        end_node = data.metadata["end"]
        optimal_cost = data.metadata["optimal_cost"]

        # Проверка старта и конца
        if pred_path[0] != start_node or pred_path[-1] != end_node:
            continue  # Путь не ведет от старта к финишу

        # Проверка ребер и расчет стоимости
        G = nx.from_numpy_array(matrix)
        path_cost = 0
        is_valid = True

        for i in range(len(pred_path) - 1):
            u, v = pred_path[i], pred_path[i + 1]
            if G.has_edge(u, v):
                path_cost += G[u][v]["weight"]
            else:
                is_valid = False
                break

        if is_valid:
            metrics["valid_path"] += 1
            gap = path_cost - optimal_cost
            metrics["avg_optimality_gap"].append(gap)

            if gap <= 0:
                metrics["optimal_count"] += 1

    # 4. Агрегация метрик
    total = metrics["total"]
    if total == 0:
        return {}

    results = {
        "accuracy": metrics["valid_path"] / total,
        "format_compliance": 1.0 - (metrics["format_error"] / total),
        "optimal_rate": metrics["optimal_count"] / total,
        "avg_optimality_gap": (
            float(np.mean(metrics["avg_optimality_gap"])) if metrics["avg_optimality_gap"] else 0.0
        ),
        "avg_reasoning_len": float(np.mean(metrics["avg_reasoning_len"])),
    }

    return results
