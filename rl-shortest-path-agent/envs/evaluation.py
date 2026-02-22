import re
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from envs.shortest_path import ShortestPathDataset
from envs.prompts import SYSTEM_PROMPT


def extract_xml_tag(text: str, tag: str) -> str:
    """Извлекает содержимое тегов <tag>...</tag>."""
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def evaluate_agent_new(model, tokenizer, dataset, device="cuda", batch_size=8, generate_kwargs=None):
    if isinstance(dataset, str):
        dataset = ShortestPathDataset.load(dataset)

    if generate_kwargs is None:
        generate_kwargs = {"max_new_tokens": 512, "temperature": 0.0, "do_sample": False}

    model.eval()

    # --- НАСТРОЙКА ТОКЕНИЗАТОРА ДЛЯ БАТЧЕЙ ---
    # Для генерации паддинг строго должен быть слева
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # -----------------------------------------

    metrics = {
        "total": 0,
        "format_error": 0,
        "hallucination_error": 0,
        "wrong_start_end_error": 0,
        "valid_path": 0,
        "optimal_count": 0,
        "avg_optimality_gap": [],
        "avg_reasoning_len": [],
    }

    # Идем по датасету шагами размера batch_size
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating (Batched)"):
        batch_data = dataset[i : i + batch_size]
        metrics["total"] += len(batch_data)

        # 1. Формируем батч промптов
        prompts = [f"{SYSTEM_PROMPT}\n\nUser: {d.question}\n\nAssistant:" for d in batch_data]

        # Токенизируем батч (добавляем padding=True)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        # 2. Батч-генерация
        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)

        # Обрезаем промпт. input_ids.shape[1] - это длина промпта (уже выровненная паддингом)
        gen_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        # 3. Парсинг и валидация ответов из батча
        for data, response in zip(batch_data, responses):
            reasoning = extract_xml_tag(response, "reasoning")
            answer = extract_xml_tag(response, "answer")
            metrics["avg_reasoning_len"].append(len(reasoning))

            if not answer:
                metrics["format_error"] += 1
                continue

            try:
                path = [int(x.strip()) for x in re.sub(r"[^\d,]", "", answer).split(",") if x.strip()]
                if not path:
                    raise ValueError
            except ValueError:
                metrics["format_error"] += 1
                continue

            # Валидация пути
            G = nx.from_numpy_array(data.metadata["matrix"])
            start, end = data.metadata["start"], data.metadata["end"]
            opt_cost = data.metadata["optimal_cost"]

            current_cost = 0
            is_broken = False

            for j in range(len(path) - 1):
                if G.has_edge(path[j], path[j + 1]):
                    current_cost += G[path[j]][path[j + 1]]["weight"]
                else:
                    is_broken = True
                    break

            # Распределение ошибок и успехов
            if is_broken:
                metrics["hallucination_error"] += 1
            elif path[0] != start or path[-1] != end:
                metrics["wrong_start_end_error"] += 1
            else:
                metrics["valid_path"] += 1
                gap = current_cost - opt_cost
                metrics["avg_optimality_gap"].append(gap)

                if gap <= 0:
                    metrics["optimal_count"] += 1

    # 5. Итоговый расчет (без изменений)
    total = metrics["total"] or 1
    return {
        "accuracy": metrics["valid_path"] / total,
        "format_compliance": 1.0 - (metrics["format_error"] / total),
        "hallucination_rate": metrics["hallucination_error"] / total,
        "wrong_endpoint_rate": metrics["wrong_start_end_error"] / total,
        "optimal_rate": metrics["optimal_count"] / total,
        "avg_optimality_gap": (
            float(np.mean(metrics["avg_optimality_gap"])) if metrics["avg_optimality_gap"] else 0.0
        ),
        "avg_reasoning_len": float(np.mean(metrics["avg_reasoning_len"])) if metrics["avg_reasoning_len"] else 0.0,
    }
