import re
import networkx as nx
import numpy as np


def extract_xml_answer(text: str) -> str:
    """Извлекает содержимое тега <answer>"""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: если тегов нет, ищем паттерн "число, число, ..."
    if text:
        candidates = re.findall(r"\d+(?:\s*,\s*\d+)+", text)
        if candidates:
            return candidates[-1]
    return ""


def correctness_reward_func(
    prompts, completions, answer, matrix, start, end, optimal_cost, **kwargs
) -> list[float]:
    """Главная функция награды с плавными градиентами (Curriculum Learning)"""
    responses = [comp[0]["content"] if isinstance(comp, list) else comp for comp in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    rewards = []

    for i in range(len(responses)):
        pred_str = extracted_responses[i]
        reward = 0.0

        if not pred_str:
            rewards.append(0.0)
            continue

        try:
            # Парсим "1, 2, 3" -> [1, 2, 3]
            # Очистка от скобок и лишних символов
            clean_str = re.sub(r"[^\d,]", "", pred_str)
            pred_path = [int(x.strip()) for x in clean_str.split(",") if x.strip()]

            if not pred_path:
                rewards.append(0.0)
                continue

            # 0. Награда за формат (модель выдала список чисел)
            reward += 0.05

            graph = nx.from_numpy_array(np.array(matrix[i]))
            actual_start, actual_end, target_cost = start[i], end[i], optimal_cost[i]

            # 1. Утешительные баллы за правильный старт и финиш
            if pred_path[0] == actual_start:
                reward += 0.05
            if pred_path[-1] == actual_end:
                reward += 0.05

            valid_transitions = 0
            actual_cost = 0

            if len(pred_path) > 1:
                # 2. Балл за движение по реальным ребрам графа
                for j in range(len(pred_path) - 1):
                    u, v = pred_path[j], pred_path[j + 1]
                    if graph.has_edge(u, v):
                        valid_transitions += 1
                        actual_cost += graph[u][v]["weight"]

                transition_ratio = valid_transitions / (len(pred_path) - 1)
                reward += 0.1 * transition_ratio

                # 3. Маршрут корректно соединяет старт и финиш
                is_complete_valid_path = (
                    pred_path[0] == actual_start
                    and pred_path[-1] == actual_end
                    and valid_transitions == (len(pred_path) - 1)
                )

                if is_complete_valid_path:
                    reward += 1.0

                    # 4. Награда за оптимальность стоимости
                    if actual_cost == target_cost:
                        reward += 2.0  # Идеальный путь!
                    elif actual_cost > 0:
                        # Если путь длиннее нужного, даем частичный балл.
                        cost_ratio = target_cost / actual_cost
                        reward += 1.0 * min(1.0, cost_ratio)

        except Exception:
            pass  # Не смогли распарсить или ошибка графа — оставляем текущий reward

        rewards.append(reward)

    # Логирование для дебага
    print("-" * 20)
    print(f"Target path: {answer[0]} | Target cost: {optimal_cost[0]}")
    print(f"Response: {extracted_responses[0]}")
    print(f"Reward: {rewards[0]:.3f}")

    return rewards


def path_format_reward_func(completions, **kwargs) -> list[float]:
    """Заменяет int_reward_func. Проверяет, что ответ это числа через запятую"""
    responses = [comp[0]["content"] if isinstance(comp, list) else comp for comp in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    pattern = re.compile(r"^\s*\d+(\s*,\s*\d+)*\s*$")
    return [0.5 if r and pattern.match(r) else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Проверяет точное соблюдение структуры тегов"""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [comp[0]["content"] if isinstance(comp, list) else comp for comp in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Проверяет наличие тегов (менее строго)"""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [comp[0]["content"] if isinstance(comp, list) else comp for comp in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    """Мягко штрафует/поощряет за частичное наличие тегов"""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [comp[0]["content"] if isinstance(comp, list) else comp for comp in completions]
    return [count_xml(c) for c in contents]
