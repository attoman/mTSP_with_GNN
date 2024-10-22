# utils/calculations.py

import torch
import torch.nn.functional as F

def calculate_distance(mission1, mission2):
    """두 미션 간의 유클리드 거리를 계산합니다."""
    return torch.sqrt(torch.sum((mission1 - mission2) ** 2, dim=-1))

def calculate_travel_time(distance, speed):
    """거리와 속도를 기반으로 이동 시간을 계산합니다."""
    return distance / (speed + 1e-5)

def calculate_cost_matrix(uav_positions, mission_coords, speeds):
    """
    거리와 UAV 속도를 기반으로 비용 행렬을 계산합니다.
    """
    num_uavs = uav_positions.size(0)
    num_missions = mission_coords.size(0)
    
    cost_matrix = torch.zeros((num_uavs, num_missions), device=uav_positions.device)
    
    for i in range(num_uavs):
        for j in range(num_missions):
            distance = calculate_distance(uav_positions[i], mission_coords[j])
            cost_matrix[i, j] = distance / (speeds[i] + 1e-5)
    
    return cost_matrix

def calculate_arrival_times(uav_positions, mission_coords, speeds):
    """
    거리와 UAV 속도를 기반으로 도착 시간을 계산합니다.
    """
    num_uavs = uav_positions.size(0)
    num_missions = mission_coords.size(0)
    
    arrival_times = torch.zeros((num_uavs, num_missions), device=uav_positions.device)
    
    for i in range(num_uavs):
        for j in range(num_missions):
            distance = calculate_distance(uav_positions[i], mission_coords[j])
            arrival_times[i, j] = distance / (speeds[i] + 1e-5)
    
    return arrival_times
