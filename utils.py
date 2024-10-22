# utils.py

import torch
import torch.nn.functional as F
import random
import numpy as np
import os
import matplotlib.pyplot as plt

def calculate_distance(mission1, mission2):
    return torch.sqrt(torch.sum((mission1 - mission2) ** 2))

def calculate_travel_time(distance, speed):
    return distance / speed

def create_edge_index(num_missions, num_uavs):
    """
    각 UAV에 대해 미션 간의 모든 가능한 경로를 연결하는 edge_index를 생성합니다.
    """
    edge_index = []
    for u in range(num_uavs):
        base = u * num_missions
        for m1 in range(num_missions):
            for m2 in range(num_missions):
                if m1 != m2:
                    edge_index.append([base + m1, base + m2])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

def create_action_mask(state):
    visited = state['visited']
    reserved = state['reserved']
    action_mask = visited | reserved
    # 항상 미션 0(출발지/도착지)을 마스크 처리
    action_mask[0] = True
    return action_mask

def calculate_cost_matrix(uav_positions, mission_coords, speeds):
    num_uavs = uav_positions.size(0)
    num_missions = mission_coords.size(0)
    
    cost_matrix = torch.zeros((num_uavs, num_missions), device=uav_positions.device)
    
    for i in range(num_uavs):
        for j in range(num_missions):
            distance = calculate_distance(uav_positions[i], mission_coords[j])
            cost_matrix[i, j] = distance / (speeds[i] + 1e-5)
    
    return cost_matrix

def calculate_arrival_times(uav_positions, mission_coords, speeds):
    num_uavs = uav_positions.size(0)
    num_missions = mission_coords.size(0)
    
    arrival_times = torch.zeros((num_uavs, num_missions), device=uav_positions.device)
    
    for i in range(num_uavs):
        for j in range(num_missions):
            distance = calculate_distance(uav_positions[i], mission_coords[j])
            arrival_times[i, j] = distance / (speeds[i] + 1e-5)
    
    return arrival_times

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def calculate_total_distance(path, missions):
    distance = 0.0
    for i in range(len(path) - 1):
        mission_from = missions[path[i]]
        mission_to = missions[path[i + 1]]
        distance += calculate_distance(mission_from, mission_to).item()
    return distance

def two_opt(path, missions):
    """
    2-opt 알고리즘을 사용하여 경로 최적화.
    시작과 끝은 고정하고, 중간 미션들만 최적화.
    
    Args:
        path (list): 현재 경로 (mission indices)
        missions (torch.Tensor): 미션 좌표
    
    Returns:
        list: 최적화된 경로
    """
    best = path.copy()
    improved = True
    best_distance = calculate_total_distance(best, missions)
    
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):  # 마지막 미션은 고정
                if j - i == 1:
                    continue
                new_path = best[:i] + best[i:j][::-1] + best[j:]
                new_distance = calculate_total_distance(new_path, missions)
                if new_distance < best_distance:
                    best = new_path
                    best_distance = new_distance
                    improved = True
        if improved:
            break
    return best

def visualize_results(env, save_path, reward=None, epsilon=None, policy_loss=None, value_loss=None, folder_name=None):
    plt.figure(figsize=(10, 10))
    missions = env.missions.cpu().numpy()
    plt.scatter(missions[:, 0], missions[:, 1], c='blue', marker='o', label='Mission')
    plt.scatter(missions[0, 0], missions[0, 1], c='green', marker='s', s=100, label='Start/End Point')
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, path in enumerate(env.paths):
        path_coords = missions[path]
        color = colors[i % len(colors)]
        plt.plot(path_coords[:, 0], path_coords[:, 1], marker='x', color=color, label=f'UAV {i} Path (Velocity: {env.speeds[i].item():.2f})')
    
    plt.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'UAV MTSP - {folder_name}')
    
    # 텍스트 추가
    text = ""
    if reward is not None:
        text += f"Reward: {reward:.2f}\n"
    if epsilon is not None:
        text += f"Epsilon: {epsilon:.4f}\n"
    if policy_loss is not None:
        text += f"Policy Loss: {policy_loss:.4f}\n"
    if value_loss is not None:
        text += f"Value Loss: {value_loss:.4f}\n"
    
    if text:
        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
