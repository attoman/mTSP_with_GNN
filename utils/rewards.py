# utils/rewards.py

def compute_reward_max_time(env):
    """
    최대 소요 시간을 최소화하는 보상 함수.
    """
    max_travel_time = env.cumulative_travel_times.max()
    reward = -max_travel_time  # 패널티로 적용
    return reward

def compute_reward_total_time(env):
    """
    전체 소요 시간 합을 최소화하는 보상 함수.
    """
    total_travel_time = env.cumulative_travel_times.sum()
    reward = -total_travel_time  # 패널티로 적용
    return reward

def compute_reward_mixed(env, alpha=0.5, beta=0.5):
    """
    최대 소요 시간과 전체 소요 시간 합을 모두 고려하는 혼합 보상 함수.
    """
    max_travel_time = env.cumulative_travel_times.max()
    total_travel_time = env.cumulative_travel_times.sum()
    reward = -(alpha * max_travel_time + beta * total_travel_time)
    return reward
