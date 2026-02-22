import re
import networkx as nx
import numpy as np


def extract_xml_answer(text: str) -> str:
    """Извлекает содержимое тега <answer>"""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def correctness_reward_func(
    prompts, completions, answer, matrix, start, end, optimal_cost, **kwargs
) -> list[float]:
    """
    Улучшенная функция награды: плотные стимулы (Dense Rewards)
    и жесткие штрафы за галлюцинации.
    """
    responses = [comp[0]["content"] if isinstance(comp, list) else comp for comp in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = []

    for i in range(len(responses)):
        pred_str = extracted_responses[i]
        adj = np.array(matrix[i])
        actual_start, actual_end, target_cost = start[i], end[i], optimal_cost[i]

        reward = 0.0

        try:
            # 1. Парсинг и базовая проверка формата
            clean_str = re.sub(r"[^\d,]", "", pred_str)
            pred_path = [int(x.strip()) for x in clean_str.split(",") if x.strip()]

            if not pred_path:
                rewards.append(0.0)
                continue

            # Награда за соблюдение структуры (числа через запятую)
            reward += 0.1

            # 2. Проверка на циклы (очень важно для графов)
            if len(pred_path) != len(set(pred_path)):
                reward -= 0.4  # Штраф за петли

            # 3. Проверка старта и финиша
            if pred_path[0] == actual_start:
                reward += 0.2
            if pred_path[-1] == actual_end:
                reward += 0.2

            # 4. Проверка валидности ребер (Галлюцинации)
            valid_steps = 0
            current_cost = 0
            hallucinated_edges = 0

            for j in range(len(pred_path) - 1):
                u, v = pred_path[j], pred_path[j + 1]
                # Проверяем, что узлы в границах и ребро существует
                if u < len(adj) and v < len(adj) and adj[u][v] > 0:
                    valid_steps += 1
                    current_cost += adj[u][v]
                else:
                    hallucinated_edges += 1

            # Если модель выдумала путь — сильно режем награду
            if hallucinated_edges > 0:
                reward -= 0.5 * hallucinated_edges
            elif len(pred_path) > 1:
                # Награда за связность (какую часть пути модель прошла честно)
                connectivity_ratio = valid_steps / (len(pred_path) - 1)
                reward += 0.5 * connectivity_ratio

            # 5. Награда за достижение цели и оптимальность
            is_complete_valid_path = (
                hallucinated_edges == 0 and pred_path[0] == actual_start and pred_path[-1] == actual_end
            )

            if is_complete_valid_path:
                reward += 1.0  # Базовая награда за найденный путь

                if current_cost == target_cost:
                    reward += 2.0  # Бинго! Кратчайший путь
                else:
                    # Плавная награда: чем ближе к идеалу, тем лучше
                    # Используем возведение в квадрат для "заострения" градиента
                    optimality_ratio = target_cost / current_cost
                    reward += 1.0 * (optimality_ratio**2)

        except Exception:
            reward = 0.0

        # Награда не должна быть отрицательной для стабильности RL
        rewards.append(max(0.0, reward))

    # Лог для первой реплики в батче
    print(
        f"Path: {extracted_responses[0]} | Target: {target_cost} | Real: {current_cost if 'current_cost' in locals() else 'N/A'} | Reward: {rewards[0]:.3f}"
    )
    return rewards


def reasoning_quality_reward_func(completions, matrix, **kwargs) -> list[float]:
    """
    Проверяет, что модель в рассуждениях упоминает веса из матрицы.
    Это мешает модели писать "бессвязный" CoT.
    """
    responses = [comp[0]["content"] if isinstance(comp, list) else comp for comp in completions]
    rewards = []

    for i, text in enumerate(responses):
        reasoning_part = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
        if not reasoning_part:
            rewards.append(0.0)
            continue

        reasoning_text = reasoning_part.group(1)
        adj = np.array(matrix[i])

        # Получаем все уникальные веса из текущего графа
        true_weights = set(adj[adj > 0].flatten().astype(int).astype(str))
        # Ищем все числа в тексте рассуждений
        found_numbers = set(re.findall(r"\d+", reasoning_text))

        if not true_weights:
            rewards.append(0.0)
            continue

        # Считаем пересечение: сколько реальных весов упомянуто
        matches = true_weights.intersection(found_numbers)
        coverage = len(matches) / min(len(true_weights), 5)  # Нам достаточно 5 упоминаний

        rewards.append(min(0.5, coverage * 0.5))

    return rewards


def format_reward_func(completions, **kwargs) -> list[float]:
    """Проверка строгого соответствия XML формату"""
    responses = [comp[0]["content"] if isinstance(comp, list) else comp for comp in completions]
    rewards = []

    # Регулярка для проверки наличия обоих тегов
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"

    for r in responses:
        score = 0.0
        if re.search(pattern, r, re.DOTALL):
            score += 0.3
        if r.startswith("<reasoning>") and "</answer>" in r:
            score += 0.2
        rewards.append(score)
    return rewards
