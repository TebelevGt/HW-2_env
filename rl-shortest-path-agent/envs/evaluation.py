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


def evaluate_agent(model, tokenizer, dataset, device="cuda", batch_size=8, generate_kwargs=None, n_samples=1):
    """Функция для оценки качества обученной модели model на выбранном датасете dataset"""
    if isinstance(dataset, str):
        dataset = ShortestPathDataset.load(dataset)

    # ВСТАВКА: Динамические параметры генерации для pass@k
    if generate_kwargs is None:
        if n_samples > 1:
            generate_kwargs = {
                "max_new_tokens": 512,
                "temperature": 0.7,  # Добавляем температуру для разнообразия
                "do_sample": True,  # Включаем сэмплирование
                "num_return_sequences": n_samples,  # 128 сэмплов
            }
        else:
            generate_kwargs = {"max_new_tokens": 512, "temperature": 0.0, "do_sample": False}

    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    metrics = {
        "total": 0,
        "format_error": 0,
        "hallucination_error": 0,
        "wrong_start_end_error": 0,
        "valid_path": 0,
        "optimal_count": 0,
        "pass_at_k": 0,  # ВСТАВКА: Счётчик успешных промптов для pass@k
        "avg_optimality_gap": [],
        "avg_reasoning_len": [],
    }

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating (Batched)"):
        batch_data = dataset[i : i + batch_size]
        metrics["total"] += len(batch_data)

        prompts = [f"{SYSTEM_PROMPT}\n\nUser: {d.question}\n\nAssistant:" for d in batch_data]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)

        gen_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        # ВСТАВКА: Группируем ответы по промптам
        for j, data in enumerate(batch_data):
            item_responses = responses[j * n_samples : (j + 1) * n_samples]
            prompt_passed = False

            # Проверяем все сэмплы для конкретного промпта
            for response in item_responses:
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

                for k in range(len(path) - 1):
                    if G.has_edge(path[k], path[k + 1]):
                        current_cost += G[path[k]][path[k + 1]]["weight"]
                    else:
                        is_broken = True
                        break

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
                        prompt_passed = True  # Нашли хотя бы один оптимальный путь!

            # ВСТАВКА: Если хотя бы 1 из 128 сэмплов был оптимальным, засчитываем pass@k
            if prompt_passed:
                metrics["pass_at_k"] += 1

    # ВСТАВКА: Корректируем делитель для метрик (ошибки считаются по всем сэмплам, pass@k - по промптам)
    total_samples = (metrics["total"] or 1) * n_samples
    total_prompts = metrics["total"] or 1

    result = {
        "accuracy": metrics["valid_path"] / total_samples,
        "format_compliance": 1.0 - (metrics["format_error"] / total_samples),
        "hallucination_rate": metrics["hallucination_error"] / total_samples,
        "wrong_endpoint_rate": metrics["wrong_start_end_error"] / total_samples,
        "optimal_rate": metrics["optimal_count"] / total_samples,
        f"pass@{n_samples}": metrics["pass_at_k"] / total_prompts,  # Вывод pass@128
        "avg_optimality_gap": (
            float(np.mean(metrics["avg_optimality_gap"])) if metrics["avg_optimality_gap"] else 0.0
        ),
        "avg_reasoning_len": float(np.mean(metrics["avg_reasoning_len"])) if metrics["avg_reasoning_len"] else 0.0,
    }
    return result
