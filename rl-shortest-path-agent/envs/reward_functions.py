import re
import networkx as nx
import numpy as np


def extract_xml_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def correctness_reward_func(
    prompts, completions, answer, matrix, start, end, optimal_cost, **kwargs
) -> list[float]:
    responses = [comp[0]["content"] if isinstance(comp, list) else comp for comp in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = []

    for i in range(len(responses)):
        pred_str = extracted_responses[i]
        adj = np.array(matrix[i])
        G = nx.from_numpy_array(adj)
        actual_start, actual_end, target_cost = start[i], end[i], optimal_cost[i]

        reward = 0.0

        try:
            # Очистка и парсинг
            clean_str = re.sub(r"[^\d,]", "", pred_str)
            pred_path = [int(x.strip()) for x in clean_str.split(",") if x.strip()]

            if not pred_path:
                rewards.append(0.0)
                continue

            # 1. Базовый формат
            reward += 0.1

            # 2. Старт должен быть верным (обязательно)
            if pred_path[0] != actual_start:
                rewards.append(0.0)  # Если старт не тот, дальше не смотрим
                continue
            reward += 0.2

            # 3. Проверка связности и накопление стоимости
            current_cost = 0
            is_broken = False
            last_valid_node = actual_start

            for j in range(len(pred_path) - 1):
                u, v = pred_path[j], pred_path[j + 1]
                if G.has_edge(u, v):
                    current_cost += G[u][v]["weight"]
                    last_valid_node = v
                else:
                    is_broken = True
                    break  # Путь прерван галлюцинацией

            # 4. "Эвристический компас" (Награда за приближение к цели)
            if not is_broken:
                try:
                    # Считаем, сколько шагов осталось от последней точки до финиша
                    dist_to_go = nx.shortest_path_length(G, source=last_valid_node, target=actual_end)
                    max_dist = len(G.nodes)
                    # Чем меньше dist_to_go, тем больше этот бонус (от 0 до 0.5)
                    reward += 0.5 * (1.0 - (dist_to_go / max_dist))
                except nx.NetworkXNoPath:
                    pass

            # 5. Бонус за успешный финиш
            if not is_broken and pred_path[-1] == actual_end:
                reward += 3.5  # Мы дошли! (Награда за выполнение задачи)
                # print(f"SUCCESS: Start: {actual_start} | End: {actual_end} | Path: {pred_path} | Valid: True")

            # Штраф за циклы, чтобы не накручивали прогресс
            if len(pred_path) != len(set(pred_path)):
                reward *= 0.5

        except Exception:
            reward = 0.0

        rewards.append(max(0.0, reward))

    # Печать для отладки
    if extracted_responses and len(answer) > 0:
        pass
        # print(
        #     f"Target: {answer[0]} | Start: {start[0]} | End: {end[0]} | Response: {extracted_responses[0]} | Reward: {rewards[0]:.3f}"
        # )

    return rewards


def reasoning_length_reward_func(completions, **kwargs) -> list[float]:
    """
    Награда за 'глубину размышлений'.
    Если модель пишет мало в <reasoning>, она скорее всего халтурит.
    """
    responses = [comp[0]["content"] if isinstance(comp, list) else comp for comp in completions]
    rewards = []
    for r in responses:
        reasoning = re.search(r"<reasoning>(.*?)</reasoning>", r, re.DOTALL)
        if reasoning:
            content = reasoning.group(1).strip()
            # Даем до 0.5 баллов за подробные рассуждения (более 300 символов)
            score = min(0.5, len(content) / 600)
            rewards.append(score)
        else:
            rewards.append(0.0)
    return rewards


def format_reward_func(completions, **kwargs) -> list[float]:
    """Проверка XML и отсутствия лишнего мусора после </answer>"""
    responses = [comp[0]["content"] if isinstance(comp, list) else comp for comp in completions]
    rewards = []
    for r in responses:
        score = 0.0
        if "<reasoning>" in r and "</reasoning>" in r:
            score += 0.15
        if "<answer>" in r and "</answer>" in r:
            score += 0.15
        # Штраф за текст после закрывающего тега (модель любит болтать)
        if r.strip().endswith("</answer>"):
            score += 0.2
        rewards.append(score)
    return rewards
