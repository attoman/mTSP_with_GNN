import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import json
import numpy as np
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm
import wandb
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# ============================
# 유틸리티 함수
# ============================

def calculate_distance(mission1, mission2):
    """두 미션 간의 유클리드 거리를 계산합니다."""
    return torch.sqrt(torch.sum((mission1 - mission2) ** 2))

def calculate_travel_time(distance, speed):
    """거리와 속도를 기반으로 이동 시간을 계산합니다."""
    return distance / speed

def create_edge_index(num_missions, num_uavs):
    """
    num_uavs와 num_missions에 기반하여 동적으로 엣지를 생성
    
    Args:
        num_missions (int): 미션의 수.
        num_uavs (int): UAV의 수.
        
    Returns:
        torch.Tensor: edge_index 텐서.
    """
    edge_index = []
    for u in range(num_uavs):
        base = u * num_missions
        for m1 in range(num_missions):
            for m2 in range(num_missions):
                if m1 != m2:
                    edge_index.append([base + m1, base + m2])
    if len(edge_index) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

def create_action_mask(state, done=False):
    """
    각 UAV에 대해 개별적인 액션 마스크를 생성합니다.
    
    Args:
        state (dict): 현재 상태로 'visited', 'reserved', 'ready_for_next_action' 텐서를 포함합니다.
        done (bool): 모든 임무가 완료되었는지 여부.
        
    Returns:
        torch.Tensor: 각 UAV에 대한 액션 마스크 텐서. (num_uavs, num_missions)
    """
    visited = state['visited']  # (num_missions,)
    reserved = state['reserved']  # (num_missions,)
    ready = state['ready_for_next_action']  # (num_uavs,)

    num_uavs = ready.size(0)
    num_missions = visited.size(0)

    # 기본적으로 visited 또는 reserved인 임무를 마스킹
    action_mask = visited.unsqueeze(0).repeat(num_uavs, 1) | reserved.unsqueeze(0).repeat(num_uavs, 1)

    # 마지막 임무(시작/종료 지점)를 마스킹
    action_mask[:, -1] = True

    # 모든 중간 임무가 완료되었을 때만 마지막 임무로 복귀를 허용
    if visited[1:-1].all().item():
        action_mask[:, -1] = False

    # UAV가 준비되지 않은 경우 모든 액션을 마스킹
    action_mask[~ready] = True  # ready가 False인 UAV는 모든 임무를 마스킹

    return action_mask  # (num_uavs, num_missions)

def calculate_cost_matrix(uav_positions, mission_coords, speeds):
    num_uavs = uav_positions.size(0)
    num_missions = mission_coords.size(0)
    dist_matrix = torch.zeros((num_uavs, num_missions), device=uav_positions.device)
    timetogo_matrix = torch.zeros((num_uavs, num_missions), device=uav_positions.device)
    
    for i in range(num_uavs):
        for j in range(num_missions):
            dist = calculate_distance(uav_positions[i], mission_coords[j])
            dist_matrix[i, j] = dist
            timetogo_matrix[i, j] = dist / (speeds[i] + 1e-5)  # 분모에 작은 값 추가
    
    # NaN 검사
    dist_matrix = torch.nan_to_num(dist_matrix)
    timetogo_matrix = torch.nan_to_num(timetogo_matrix)

    return timetogo_matrix, dist_matrix

# ============================
# 보상 함수 구현
# ============================

def compute_reward_max_time(env, max_possible_time=1000, use_2opt=True):
    try:
        if use_2opt:
            optimized_paths = [apply_2opt(path, env.missions) for path in env.paths]
            optimized_travel_times = calculate_total_travel_times(optimized_paths, env.missions, env.speeds)
        else:
            optimized_travel_times = env.cumulative_travel_times

        max_travel_time = optimized_travel_times.max().item()

        if torch.isnan(torch.tensor(max_travel_time)) or max_travel_time == 0:
            return 0.0

        reward = max_possible_time / (1 + max_travel_time)  # 최대 이동 시간이 낮을수록 높은 보상
        return reward
    except Exception as e:
        print(f"compute_reward_max_time에서 오류 발생: {e}")
        return 0.0


def compute_reward_total_time(env, max_possible_time=1000, use_2opt=True):
    try:
        if use_2opt:
            optimized_paths = [apply_2opt(path, env.missions) for path in env.paths]
            optimized_travel_times = calculate_total_travel_times(optimized_paths, env.missions, env.speeds)
        else:
            optimized_travel_times = env.cumulative_travel_times

        total_travel_time = optimized_travel_times.sum().item()

        if torch.isnan(torch.tensor(total_travel_time)) or total_travel_time == 0:
            return 0.0

        reward = max_possible_time / (1 + total_travel_time)  # 총 이동 시간이 낮을수록 높은 보상
        return reward
    except Exception as e:
        print(f"compute_reward_total_time에서 오류 발생: {e}")
        return 0.0

def compute_reward_mixed(env, alpha=0.5, beta=0.3, gamma=0.1, max_possible_time=1000, use_2opt=True):
    """
    최대, 총, 이동 시간 분산을 혼합하여 보상 계산, 선택적으로 3-opt 적용

    Args:
        env (MissionEnvironment): 환경 인스턴스
        alpha (float): 최대 이동 시간 가중치
        beta (float): 총 이동 시간 가중치
        gamma (float): 이동 시간 분산 가중치
        max_possible_time (float): 스케일링을 위한 최대 가능 시간
        use_2opt (bool): 3-opt 최적화를 적용할지 여부

    Returns:
        float: 보상 값
    """
    try:
        if use_2opt:
            optimized_paths = [apply_2opt(path, env.missions) for path in env.paths]
            optimized_travel_times = calculate_total_travel_times(optimized_paths, env.missions, env.speeds)
        else:
            optimized_travel_times = env.cumulative_travel_times

        max_travel_time = optimized_travel_times.max().item()
        total_travel_time = optimized_travel_times.sum().item()
        time_variance = optimized_travel_times.var().item()
        
        # 시간 효율성 지표 추가
        time_efficiency = torch.mean(1 / (optimized_travel_times + 1e-5))
        
        # UAV 간 작업 균형 지표
        workload_balance = torch.exp(-time_variance)
        mission_distribution = torch.tensor([len(path) for path in env.paths]).float()
        workload_distribution = 1 / (mission_distribution.var() + 1e-5)
        
        # 복합 보상 계산
        weighted_time = (
            alpha * (max_possible_time / (1 + total_travel_time)) +          # 총 시간 최소화
            beta * time_efficiency +        # 총 시간 최소화
            gamma * workload_balance  +          # 시간 효율성
            0.1 * workload_distribution  # UAV 간 작업 균형
        )

        base_reward = max_possible_time / (1 + weighted_time)
        
        # # 추가 보너스/페널티
        # completion_bonus = 1.0 if env.visited[1:-1].all() else 0.0
        return base_reward
    
    except Exception as e:
        print(f"compute_reward_mixed에서 오류 발생: {e}")
        return max_possible_time

def compute_episode_reward(env, reward_type='total', alpha=0.5, beta=0.5, gamma=0.5, max_possible_time=1000, use_2opt=True):
    if reward_type == 'max':
        return compute_reward_max_time(env, max_possible_time, use_2opt)
    elif reward_type == 'total':
        return compute_reward_total_time(env, max_possible_time, use_2opt)
    elif reward_type == 'mixed':
        return compute_reward_mixed(env, alpha, beta, gamma, max_possible_time, use_2opt)
    else:
        raise ValueError(f"알 수 없는 reward_type: {reward_type}")


def clip_rewards(rewards, min_value=-1000, max_value=1000):
    """
    보상을 특정 범위로 제한하여 정규화 대신 클리핑 방식으로 보상을 조절합니다.
    """
    rewards = torch.tensor(rewards, dtype=torch.float32)
    return torch.clamp(rewards, min=min_value, max=max_value)

def choose_action(action_logits, dist_matrix, timetogo_matrix, temperature, uav_order, global_action_mask=None):
    """
    가변 UAV 및 미션 수에 대응하는 유연한 액션 선택 함수.
    """
    num_uavs, num_missions = action_logits.shape
    actions = [-1] * num_uavs  # 초기화

    for i in uav_order:
        if global_action_mask is not None:
            available_actions = (global_action_mask[i] == 0).nonzero(as_tuple=True)[0].tolist()
            if not available_actions:
                continue  # 선택 가능한 액션이 없으면 스킵

            logits_i = action_logits[i, available_actions]
            distances = dist_matrix[i, available_actions]
            times = timetogo_matrix[i, available_actions]
            
            """ # 거리 기반으로 로그 확률 조정
            mean_distance = distances.mean()
            std_distance = distances.std() + 1e-5  # 0으로 나누는 것을 방지
            z_scores = (distances - mean_distance) / std_distance
            weighted_logits = logits_i - z_scores
            probs_i = F.softmax(weighted_logits, dim=-1).detach().cpu().numpy() """
            
            # 시간 기반으로 로그 확률 조정
            mean_time = times.mean()
            std_time = times.std() + 1e-5  # 0으로 나누는 것을 방지
            z_scores = (times - mean_time) / std_time
            weighted_logits = logits_i - z_scores
            probs_i = F.softmax(weighted_logits, dim=-1).detach().cpu().numpy()

            # NaN 또는 유효하지 않은 확률 처리
            if np.isnan(probs_i).any() or not np.isfinite(probs_i).all():
                chosen_action = random.choice(available_actions)
            else:
                chosen_action = np.random.choice(available_actions, p=probs_i)

            # 다른 UAV가 선택한 액션을 선택하지 못하도록 마스킹
            for j in range(num_uavs):
                if j != i:
                    global_action_mask[j, chosen_action] = True
        else:
            logits_i = action_logits[i] / temperature - dist_matrix[i] / temperature
            probs_i = F.softmax(logits_i, dim=-1).detach().cpu().numpy()
            if np.isnan(probs_i).any() or not np.isfinite(probs_i).all():
                chosen_action = random.randint(0, num_missions - 1)
            else:
                chosen_action = np.random.choice(num_missions, p=probs_i)

        actions[i] = chosen_action

    return actions

def apply_2opt(path, mission_coords):
    """
    UAV 경로를 최적화하기 위해 2-opt 알고리즘 적용

    Args:
        path (list): 현재 UAV 경로
        mission_coords (torch.Tensor): 미션 좌표들. (num_missions, 2)

    Returns:
        list: 최적화된 UAV 경로
    """
    best_path = path[:]
    best_distance = calculate_total_distance(best_path, mission_coords)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) - 1):
                if j - i == 1:  # 인접한 엣지 교환은 의미가 없으므로 건너뜀
                    continue
                new_path = path[:i] + path[i:j][::-1] + path[j:]
                new_distance = calculate_total_distance(new_path, mission_coords)
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_path = new_path
                    improved = True
        path = best_path  # 경로 업데이트
    return best_path

def calculate_total_distance(path, mission_coords):
    """
    주어진 경로의 총 거리를 계산

    Args:
        path (list): UAV 경로
        mission_coords (torch.Tensor): 미션 좌표들. (num_missions, 2)

    Returns:
        float: 총 거리
    """
    total_distance = 0.0
    for i in range(len(path) - 1):
        total_distance += calculate_distance(mission_coords[path[i]], mission_coords[path[i+1]]).item()
    return total_distance


def calculate_total_travel_times(paths, mission_coords, speeds):
    """
    UAV들의 경로에 따른 총 이동 시간을 계산합니다.
    
    Args:
        paths (list of list): 각 UAV의 경로 리스트.
        mission_coords (torch.Tensor): 미션 좌표 텐서 (num_missions, 2).
        speeds (torch.Tensor): 각 UAV의 속도 텐서 (num_uavs,).
    
    Returns:
        torch.Tensor: UAV들의 총 이동 시간.
    """
    total_travel_times = torch.zeros(len(paths))
    for i, path in enumerate(paths):
        travel_time = 0.0
        for j in range(len(path) - 1):
            distance = calculate_distance(mission_coords[path[j]], mission_coords[path[j + 1]])
            travel_time += distance / (speeds[i] + 1e-5)  # 이동 시간 계산
        total_travel_times[i] = travel_time
    return total_travel_times

def compute_uav_order(env):
    """
    준비된 UAV들만을 포함하여 UAV 선택 순서를 결정합니다.
    
    Args:
        env (MissionEnvironment): 미션 환경.
        
    Returns:
        list: 정렬된 준비된 UAV 인덱스 리스트.
    """
    # 준비된 UAV들만 선택
    ready_uavs = [i for i in range(env.num_uavs) if env.ready_for_next_action[i]]
    
    # 준비된 UAV들을 속도 기준으로 오름차순 정렬 (더 느린 UAV가 먼저 선택)
    uav_order = sorted(ready_uavs, key=lambda i: env.speeds[i].item())
    
    return uav_order


# ============================
# 특성 중요도 분석
# ============================

def compute_feature_importance(policy_net, mission_coords, edge_index, batch, uavs_info, action_mask, speeds, dist_matrix, timetogo_matrix, device):
    """
    Gradient-Based Feature Importance를 계산합니다.
    
    Args:
        policy_net (nn.Module): 정책 네트워크.
        mission_coords (torch.Tensor): 미션의 좌표. (num_missions, 2)
        edge_index (torch.Tensor): GNN을 위한 엣지 인덱스.
        batch (torch.Tensor): 배치 인덱스.
        uavs_info (torch.Tensor): UAV 위치 정보. (num_uavs, 2)
        action_mask (torch.Tensor): 액션 마스크.
        speeds (torch.Tensor): UAV 속도. (num_uavs,)
        dist_matrix (torch.Tensor): 비용 행렬. (num_uavs, num_missions)
        timetogo_matrix (torch.Tensor): 도착 시간 텐서. (num_uavs, num_missions)
        device (torch.device): 실행 장치.
        
    Returns:
        dict: 각 특성의 중요도.
    """
    
    policy_net.zero_grad()
    
    if isinstance(policy_net, nn.DataParallel):
        policy_net = policy_net.module
    
    # 입력 텐서에 기울기 계산을 위해 requires_grad 설정
    mission_coords = mission_coords.clone().detach().requires_grad_(True).to(device)  # (num_missions, 2)
    uavs_info = uavs_info.clone().detach().requires_grad_(True).to(device)          # (num_uavs, 2)
    speeds = speeds.clone().detach().requires_grad_(True).to(device)                # (num_uavs,)
    dist_matrix = dist_matrix.clone().detach().requires_grad_(True).to(device)      # (num_uavs, num_missions)
    timetogo_matrix = timetogo_matrix.clone().detach().requires_grad_(True).to(device)  # (num_uavs, num_missions)
    
    # 정책 네트워크 순전파
    action_probs, state_values = policy_net(
        mission_coords, 
        edge_index, 
        batch, 
        uavs_info, 
        action_mask, 
        speeds, 
        dist_matrix,
        timetogo_matrix
    )
    
    # 상태 가치의 평균을 스칼라 값으로 만들기
    state_values_mean = state_values.mean()
    
    # 역전파를 통해 기울기 계산
    state_values_mean.backward()
    
    # 각 입력 특성에 대한 기울기 추출 및 평균
    feature_gradients = {}
    feature_gradients["mission_x"] = mission_coords.grad[:, 0].abs().mean().item()
    feature_gradients["mission_y"] = mission_coords.grad[:, 1].abs().mean().item()
    feature_gradients["uav_x"] = uavs_info.grad[:, 0].abs().mean().item()
    feature_gradients["uav_y"] = uavs_info.grad[:, 1].abs().mean().item()
    feature_gradients["speed"] = speeds.grad.abs().mean().item()
    feature_gradients["distance"] = dist_matrix.grad.abs().mean().item()
    feature_gradients["time_to_go"] = timetogo_matrix.grad.abs().mean().item()
    
    return feature_gradients

# ============================
# 데이터 클래스
# ============================

class MissionData:
    """
    미션 데이터를 생성하고 관리하는 클래스.
    """
    def __init__(self, num_missions=20, num_uavs=3, seed=None, device='cpu'):
        self.num_missions = num_missions
        self.num_uavs = num_uavs
        self.seed = seed
        self.device = device
        self.missions, self.uavs_start, self.uavs_speeds = self.generate_data()

    def generate_data(self):
        """랜덤 미션 좌표, UAV 시작 위치, 속도를 생성합니다."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
        else:
            seed = torch.randint(0, 10000, (1,)).item()
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        missions = torch.rand((self.num_missions, 2)) * 100
        start_end_point = missions[0].clone()
        missions[-1] = start_end_point
        uavs_start = start_end_point.unsqueeze(0).repeat(self.num_uavs, 1)
        # uavs_speeds = torch.full((self.num_uavs,), 10.0)
        uavs_speeds = torch.randint(5, 30, (self.num_uavs,), dtype=torch.float)
        return missions.to(self.device), uavs_start.to(self.device), uavs_speeds.to(self.device)

    def reset_data(self, seed=None):
        """새로운 시드를 사용하여 미션 데이터를 재설정합니다."""
        self.seed = seed
        self.missions, self.uavs_start, self.uavs_speeds = self.generate_data()

# ============================
# 강화 학습 환경 클래스 (다중 에이전트)
# ============================

class MissionEnvironment:
    """
    다중 UAV 미션 할당을 위한 강화 학습 환경 클래스.
    """
    def __init__(self, missions=None, uavs_start=None, uavs_speeds=None, device='cpu', mode='train', seed=None, time_weight=2.0, use_2opt=False):
        self.device = device
        self.mode = mode
        self.seed = seed
        self.num_missions = missions.size(0) if missions is not None else 20
        self.num_uavs = uavs_start.size(0) if uavs_start is not None else 3

        self.missions = missions
        self.uavs_start = uavs_start
        self.speeds = uavs_speeds
        self.use_2opt = use_2opt  # use_2opt 인자를 저장
        self.time_weight = time_weight
        self.reset()

    def reset(self):
        """환경을 초기 상태로 재설정"""
        self.current_positions = self.uavs_start.clone()
        self.visited = torch.zeros(self.num_missions, dtype=torch.bool, device=self.device)
        self.reserved = torch.zeros(self.num_missions, dtype=torch.bool, device=self.device)
        self.paths = [[] for _ in range(self.num_uavs)]
        self.cumulative_travel_times = torch.zeros(self.num_uavs, device=self.device)
        self.ready_for_next_action = torch.ones(self.num_uavs, dtype=torch.bool, device=self.device)
        self.targets = [-1] * self.num_uavs
        self.remaining_distances = torch.full((self.num_uavs,), float('inf'), device=self.device)

        for i in range(self.num_uavs):
            self.paths[i].append(0)
        
        # get_state() 대신 get_state_enhanced() 사용
        return self.get_state_enhanced()
    
    def get_state(self):
        """
        현재 환경 상태를 반환
        Returns:
            dict: 현재 상태 정보를 담은 딕셔너리
        """
        return self.get_state_enhanced()
    
    def calculate_uav_workload(self):
        """각 UAV의 작업 부하를 계산"""
        workload = torch.zeros(self.num_uavs, device=self.device)
        
        for i in range(self.num_uavs):
            # 이동 거리 기반 작업 부하
            total_distance = sum(calculate_distance(self.missions[self.paths[i][j]], 
                                                self.missions[self.paths[i][j+1]]) 
                            for j in range(len(self.paths[i])-1))
            
            # 시간 기반 작업 부하
            time_load = self.cumulative_travel_times[i]
            
            # 미션 수 기반 작업 부하
            mission_count = len(self.paths[i])
            
            # 종합적인 작업 부하 계산
            workload[i] = (0.4 * total_distance + 
                        0.4 * time_load + 
                        0.2 * mission_count)
        
        return workload

    def calculate_time_efficiency(self):
        """각 UAV의 시간 효율성 계산"""
        efficiency = torch.zeros(self.num_uavs, device=self.device)
        
        for i in range(self.num_uavs):
            if len(self.paths[i]) > 1:
                # 이동 거리
                total_distance = sum(calculate_distance(self.missions[self.paths[i][j]], 
                                                    self.missions[self.paths[i][j+1]]) 
                                for j in range(len(self.paths[i])-1))
                
                # 평균 속도 계산
                avg_speed = total_distance / (self.cumulative_travel_times[i] + 1e-5)
                
                # 시간당 완료된 미션 수
                missions_per_time = len(self.paths[i]) / (self.cumulative_travel_times[i] + 1e-5)
                
                # 효율성 점수 계산
                efficiency[i] = (0.5 * avg_speed / self.speeds[i] + 
                            0.5 * missions_per_time)
        
        return efficiency
    
    def calculate_mission_priorities(self):
        """각 미션의 우선순위 계산"""
        priorities = torch.zeros(self.num_missions, device=self.device)
        
        for i in range(self.num_missions):
            if not self.visited[i]:
                # 거리 기반 우선순위
                min_distance = float('inf')
                for j in range(self.num_uavs):
                    if self.ready_for_next_action[j]:
                        dist = calculate_distance(self.current_positions[j], 
                                            self.missions[i])
                        min_distance = min(min_distance, dist)
                
                # 시간 기반 우선순위
                time_factor = 1.0
                if i == self.num_missions - 1:  # 시작/종료 지점
                    time_factor = 0.5 if not self.visited[1:-1].all() else 2.0
                
                # 우선순위 점수 계산
                priorities[i] = (1.0 / (min_distance + 1e-5)) * time_factor
        
        return priorities
    
    def calculate_estimated_arrival_times(self):
        """각 UAV의 현재 목표 지점까지의 예상 도착 시간 계산"""
        arrival_times = torch.zeros(self.num_uavs, device=self.device)
        
        for i in range(self.num_uavs):
            if self.targets[i] != -1 and not self.ready_for_next_action[i]:
                # 남은 거리 기반 예상 시간
                remaining_time = self.remaining_distances[i] / (self.speeds[i] + 1e-5)
                arrival_times[i] = remaining_time
        
        return arrival_times

    def get_state_enhanced(self):
    # 기본 상태 정보
        base_state = {
            'positions': self.current_positions.clone(),
            'visited': self.visited.clone(),
            'reserved': self.reserved.clone(),
            'ready_for_next_action': self.ready_for_next_action.clone(),
            
            # 시간 관련 정보 추가
            'cumulative_times': self.cumulative_travel_times.clone(),
            'estimated_arrival_times': self.calculate_estimated_arrival_times(),
            'time_efficiency': self.calculate_time_efficiency(),
            
            # UAV 상태 정보 강화
            'uav_workload': self.calculate_uav_workload(),
            'mission_priorities': self.calculate_mission_priorities(),
            'path_history': [path.copy() for path in self.paths],
            
            # 미션 완료 진행도
            'completion_rate': self.visited[1:-1].float().mean(),
            'remaining_missions': (~self.visited[1:-1]).sum().item()
        }
        return base_state

    def step(self, actions):
        """
        액션을 실행하고 환경 상태를 업데이트합니다.
        
        Args:
            actions (list): UAV들이 선택한 액션.
        
        Returns:
            tuple: 다음 상태, 소요 시간 텐서, 종료 여부.
        """
        for i, action in enumerate(actions):
            if self.ready_for_next_action[i] and not self.visited[action] and not self.reserved[action]:
                self.reserved[action] = True
                self.ready_for_next_action[i] = False
                self.targets[i] = action
                mission_from = self.current_positions[i]
                mission_to = self.missions[action]
                self.remaining_distances[i] = calculate_distance(mission_from, mission_to)

        travel_times = torch.zeros(self.num_uavs, device=self.device)

        for i, action in enumerate(self.targets):
            if action != -1 and not self.ready_for_next_action[i]:
                distance = self.remaining_distances[i]
                travel_time = calculate_travel_time(distance, self.speeds[i].item())

                self.cumulative_travel_times[i] += travel_time
                self.current_positions[i] = self.missions[action]
                self.visited[action] = True
                self.paths[i].append(action)
                self.ready_for_next_action[i] = True
                self.reserved[action] = False

                travel_times[i] = travel_time

        done = self.visited[1:-1].all()  # 마지막 미션을 제외한 모든 미션이 완료되었는지 확인

        # 2-opt 알고리즘 적용 (플래그가 활성화된 경우)
        if self.use_2opt:
            for i in range(self.num_uavs):
                self.paths[i] = apply_2opt(self.paths[i], self.missions)

        # 모든 중간 미션이 완료된 후 UAV들이 반드시 시작/종료 지점으로 돌아가도록 처리
        if done:
            for i in range(self.num_uavs):
                if self.targets[i] != self.num_missions - 1:  # 아직 시작/종료 지점에 도착하지 않은 경우
                    self.targets[i] = self.num_missions - 1
                    self.ready_for_next_action[i] = False
                    self.remaining_distances[i] = calculate_distance(self.current_positions[i], self.missions[-1])

        return self.get_state(), travel_times, done and self.visited[-1]  # 모든 UAV가 시작/종료 지점에 도착했을 때 완전히 종료

# ============================
# GNN Transformer 인코더
# ============================

class EnhancedGNNTransformerEncoder(nn.Module):
    """
    Self-attention을 포함한 GNN Transformer 인코더에 Residual Connections를 추가.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=8, heads=8, dropout=0.3):
        super(EnhancedGNNTransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.hidden_channels = hidden_channels

        self.layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        for layer in range(num_layers):
            conv = TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
            self.layers.append(conv)
            
            # Residual 프로젝션: 입력과 출력의 차원이 다르면 선형 프로젝션을 통해 맞춤
            if in_channels != hidden_channels * heads:
                projection = nn.Linear(in_channels, hidden_channels * heads)
            else:
                projection = nn.Identity()
            self.residual_projections.append(projection)
            
            in_channels = hidden_channels * heads  # 다음 레이어의 입력 채널 업데이트

        self.gnn_output = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x, edge_index, batch):
        """
        GNN Transformer 인코더를 통한 순전파.

        Args:
            x (torch.Tensor): 노드 특성. (num_nodes, in_channels)
            edge_index (torch.Tensor): 엣지 인덱스. (2, num_edges)
            batch (torch.Tensor): 배치 인덱스. (num_nodes,)

        Returns:
            torch.Tensor: 노드 임베딩. (num_nodes, out_channels)
        """
        for layer in range(self.num_layers):
            residual = self.residual_projections[layer](x)  # Residual 연결을 위한 입력 변환
            x_new = self.layers[layer](x, edge_index)  # TransformerConv 레이어
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.gnn_output[1].p, training=self.training)
            x = x_new + residual  # Residual Connection

        x = self.gnn_output(x)
        return x


# ============================
# 액터-크리틱 네트워크
# ============================


class ImprovedActorCriticNetwork(nn.Module):
    """
    Enhanced GNN Transformer 인코더를 사용하는 액터-크리틱 네트워크.
    Residual Connections가 추가된 GNN 인코더와 호환됩니다.
    """
    def __init__(self, num_missions, num_uavs, embedding_dim=64, gnn_hidden_dim=64, 
                 actor_hidden_dim=128, critic_hidden_dim=128,
                 num_layers=8, heads=8,
                 gnn_dropout=0.3, actor_dropout=0.3, critic_dropout=0.3):
        super(ImprovedActorCriticNetwork, self).__init__()
        self.num_missions = num_missions
        self.num_uavs = num_uavs

        # 총 입력 채널: 미션 좌표(2) + 마스크(1) + UAV 속도(1) + 비용(1) + 예상 도착 시간(1) = 6
        total_in_channels = 2 + 1 + 1 + 1 + 1

        # GNN Transformer Encoder with Residual Connections
        self.gnn_encoder = EnhancedGNNTransformerEncoder(
            in_channels=total_in_channels,
            hidden_channels=gnn_hidden_dim,
            out_channels=embedding_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=gnn_dropout
        )

        # Projection layers to make embed_dim + 3 divisible by num_heads
        # 예를 들어, heads=8일 때, embedding_dim + 3 = 64 (8 * 8)
        # 따라서 embedding_dim을 61로 설정 (기본값은 64로 조정 필요)
        self.actor_projection = nn.Linear(embedding_dim + 3, heads * 8)  # (num_uavs, 64)
        self.critic_projection = nn.Linear(embedding_dim + 3, heads * 8)  # (num_uavs, 64)

        # MultiheadAttention layers
        self.actor_attention = nn.MultiheadAttention(embed_dim=heads * 8, num_heads=heads, dropout=actor_dropout)
        self.actor_shared = nn.Sequential(
            nn.Linear(heads * 8, actor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(actor_dropout)
        )

        # Actor output layer
        self.actor_out = nn.Linear(actor_hidden_dim, num_missions)

        # Critic MultiheadAttention
        self.critic_attention = nn.MultiheadAttention(embed_dim=heads * 8, num_heads=heads, dropout=critic_dropout)
        self.critic_shared = nn.Sequential(
            nn.Linear(heads * 8, critic_hidden_dim),
            nn.ReLU(),
            nn.Dropout(critic_dropout)
        )

        # Critic output layer
        self.critic_out = nn.Linear(critic_hidden_dim, 1)

    def forward(self, mission_coords, edge_index, batch, uavs_info, action_mask, speeds, dist_matrix, timetogo_matrix):
        """
        액터-크리틱 네트워크를 통한 순전파.

        Args:
            mission_coords (torch.Tensor): 미션의 좌표. (num_missions, 2)
            edge_index (torch.Tensor): GNN을 위한 엣지 인덱스.
            batch (torch.Tensor): 배치 인덱스.
            uavs_info (torch.Tensor): UAV 위치 정보. (num_uavs, 2)
            action_mask (torch.Tensor): 액션 마스크. (num_uavs, num_missions)
            speeds (torch.Tensor): UAV 속도. (num_uavs,)
            dist_matrix (torch.Tensor): 거리 행렬. (num_uavs, num_missions)
            timetogo_matrix (torch.Tensor): 도착 시간 행렬. (num_uavs, num_missions)

        Returns:
            tuple: 액션 로그 확률과 상태 값.
        """

        # 각 텐서를 3차원으로 맞춥니다.
        mask_embedded = action_mask.unsqueeze(-1).float()  # (num_uavs, num_missions, 1)
        speeds_embedded = speeds.unsqueeze(-1).unsqueeze(1).repeat(1, mission_coords.size(0), 1)  # (num_uavs, num_missions, 1)
        dist_embedded = dist_matrix.unsqueeze(-1)  # (num_uavs, num_missions, 1)
        timetogo_embedded = timetogo_matrix.unsqueeze(-1)  # (num_uavs, num_missions, 1)

        # 미션 좌표를 각 UAV에 대해 확장하여 맞춥니다.
        mission_coords_expanded = mission_coords.unsqueeze(0).repeat(uavs_info.size(0), 1, 1)  # (num_uavs, num_missions, 2)

        # 텐서 결합
        combined_embedded = torch.cat([
            mission_coords_expanded,  # (num_uavs, num_missions, 2)
            mask_embedded,            # (num_uavs, num_missions, 1)
            speeds_embedded,          # (num_uavs, num_missions, 1)
            dist_embedded,            # (num_uavs, num_missions, 1)
            timetogo_embedded         # (num_uavs, num_missions, 1)
        ], dim=-1)  # (num_uavs, num_missions, 6)

        combined_embedded = combined_embedded.view(-1, combined_embedded.size(-1))  # (num_uavs * num_missions, 6)
        new_batch = batch.repeat_interleave(mission_coords.size(0))  # (num_uavs * num_missions,)

        # GNN 인코더를 통한 임베딩 생성
        mission_embeddings = self.gnn_encoder(combined_embedded, edge_index, new_batch)  # (num_uavs * num_missions, embedding_dim)
        mission_embeddings = mission_embeddings.view(uavs_info.size(0), mission_coords.size(0), -1)  # (num_uavs, num_missions, embedding_dim)
        uav_embeddings = mission_embeddings.sum(dim=1)  # (num_uavs, embedding_dim)

        # UAV의 임베딩과 추가 정보를 결합
        combined = torch.cat([
            uavs_info,          # (num_uavs, 2)
            uav_embeddings,     # (num_uavs, embedding_dim)
            speeds.unsqueeze(-1)   # (num_uavs, 1)
        ], dim=-1)  # (num_uavs, embedding_dim + 3)

        # 액터 네트워크: 선형 계층을 통한 매핑 후 어텐션
        actor_proj = self.actor_projection(combined)  # (num_uavs, 64)
        actor_proj = actor_proj.unsqueeze(1)  # (num_uavs, 1, 64)
        actor_attn_output, _ = self.actor_attention(actor_proj, actor_proj, actor_proj)
        actor_attn_output = actor_attn_output.squeeze(1)  # (num_uavs, 64)
        actor_features = self.actor_shared(actor_attn_output)  # (num_uavs, actor_hidden_dim)
        action_logits = self.actor_out(actor_features)  # (num_uavs, num_missions)

        # 크리틱 네트워크: 선형 계층을 통한 매핑 후 어텐션
        critic_proj = self.critic_projection(combined)  # (num_uavs, 64)
        critic_proj = critic_proj.unsqueeze(1)  # (num_uavs, 1, 64)
        critic_attn_output, _ = self.critic_attention(critic_proj, critic_proj, critic_proj)
        critic_attn_output = critic_attn_output.squeeze(1)  # (num_uavs, 64)
        critic_features = self.critic_shared(critic_attn_output)  # (num_uavs, critic_hidden_dim)
        state_values = self.critic_out(critic_features).squeeze()  # (num_uavs,)

        return action_logits, state_values
    
# ============================
# 학습 및 검증 함수
# ============================

def train_model(env, val_env, policy_net, optimizer_actor, optimizer_critic, scheduler_actor, scheduler_critic,
               num_epochs, batch_size, device, edge_index, batch, temperature, lr_gamma, 
               reward_type='total', alpha=0.5, beta=0.5, gamma=0.5,
               entropy_coeff=0.01,  # 기본값 설정
               start_epoch=1, checkpoint_path=None, results_path=None, checkpoints_path=None, patience=10, wandb_name="run", use_2opt=False):
    
    # WandB 초기화
    wandb.init(project="multi_uav_mission", name=wandb_name, config={
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate_actor": optimizer_actor.param_groups[0]['lr'],
        "learning_rate_critic": optimizer_critic.param_groups[0]['lr'],
        "weight_decay_actor": optimizer_actor.param_groups[0]['weight_decay'],
        "weight_decay_critic": optimizer_critic.param_groups[0]['weight_decay'],
        "lr_gamma": lr_gamma,
        "patience": patience,
        "gnn_dropout": policy_net.module.gnn_encoder.gnn_output[1].p if isinstance(policy_net, nn.DataParallel) else policy_net.gnn_encoder.gnn_output[1].p,
        "actor_dropout": policy_net.module.actor_shared[-1].p if isinstance(policy_net, nn.DataParallel) else policy_net.actor_shared[-1].p,
        "critic_dropout": policy_net.module.critic_shared[-1].p if isinstance(policy_net, nn.DataParallel) else policy_net.critic_shared[-1].p,
        "num_missions": env.num_missions,
        "num_uavs": env.num_uavs,
        "reward_type": reward_type,
        "alpha": alpha,
        "beta": beta,
        "entropy_coeff": entropy_coeff,  # 엔트로피 가중치 로깅
        "actor_layers": len(policy_net.module.actor_shared) // 3 if isinstance(policy_net, nn.DataParallel) else len(policy_net.actor_shared) // 3,
        "critic_layers": len(policy_net.module.critic_shared) // 3 if isinstance(policy_net, nn.DataParallel) else len(policy_net.critic_shared) // 3,
        "gnn_hidden_dim": policy_net.module.gnn_encoder.layers[0].out_channels // policy_net.module.gnn_encoder.layers[0].heads if isinstance(policy_net, nn.DataParallel) else policy_net.gnn_encoder.layers[0].out_channels // policy_net.gnn_encoder.layers[0].heads,
        "actor_hidden_dim": policy_net.module.actor_shared[0].in_features if isinstance(policy_net, nn.DataParallel) else policy_net.actor_shared[0].in_features,
        "critic_hidden_dim": policy_net.module.critic_shared[0].in_features if isinstance(policy_net, nn.DataParallel) else policy_net.critic_shared[0].in_features,
        "temperature": temperature  # Boltzmann 탐험 온도
    })
    
    temperature_min = 0.5
    temperature_decay = 0.995

    total_episodes = num_epochs * batch_size
    episode = (start_epoch - 1) * batch_size

    best_validation_reward = -float('inf')
    epochs_no_improve = 0
    
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        # 옵티마이저 상태를 새 GPU에 맞게 업데이트
        for optimizer in [optimizer_actor, optimizer_critic]:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        if 'temperature' in checkpoint:
            temperature = checkpoint['temperature']
        print(f"체크포인트 '{checkpoint_path}'가 로드되었습니다. 학습을 다시 시작합니다.")
    else:
        print(f"체크포인트 '{checkpoint_path}'를 입력이 없습니다. 학습을 시작합니다.")
        pass

    try:
        for epoch in tqdm(range(start_epoch, num_epochs + 1), desc="Epochs Progress", position=0):
            epoch_pbar = tqdm(range(batch_size), desc=f"에폭 {epoch}/{num_epochs}", leave=False)
            for batch_idx in epoch_pbar:
                state = env.reset()
                done = False
                log_probs = []
                values = []
                entropy_list = []
                travel_times = []

                while not done:
                    positions = state['positions']
                    uavs_info = positions.to(device)
                    # 임무 완료 여부에 따라 액션 마스크 생성
                    action_mask = create_action_mask(state, done=done)
                    
                    # 비용 행렬과 도착 시간 계산
                    timetogo_matrix, dist_matrix = calculate_cost_matrix(positions, env.missions, env.speeds)

                    # 정책 네트워크 순전파
                    action_logits, state_values = policy_net(
                        env.missions, 
                        edge_index, 
                        batch, 
                        uavs_info, 
                        action_mask, 
                        env.speeds,
                        dist_matrix,
                        timetogo_matrix,
                    )
                    
                    # UAV 선택 순서 결정 (준비된 UAV들만 포함)
                    uav_order = compute_uav_order(env)
                    
                    # 액션 선택
                    actions = choose_action(action_logits, dist_matrix, timetogo_matrix, temperature, uav_order, global_action_mask=action_mask)

                    # 각 UAV의 액션에 대한 log_prob과 state_value 수집
                    for i, action in enumerate(actions):
                        if action != -1:
                            # 선택된 액션의 확률을 가져옵니다.
                            prob = F.softmax(action_logits[i], dim=-1)[action]
                            if torch.isnan(prob):
                                print(f"NaN detected in action_probs[{i}, {action}] at Epoch {epoch}, Batch {batch_idx+1}")
                                prob = torch.tensor(1.0 / env.num_missions, device=device)
                            log_prob = torch.log(prob + 1e-10).squeeze()  # 스칼라로 만듦
                            log_probs.append(log_prob)
                            values.append(state_values[i].squeeze())  # 스칼라로 만듦

                    # 엔트로피 계산 및 저장
                    entropy = -(F.softmax(action_logits, dim=-1) * torch.log(F.softmax(action_logits, dim=-1) + 1e-10)).sum(dim=-1).mean()
                    entropy_list.append(entropy)

                    # 환경 스텝
                    next_state, travel_time, done = env.step(actions)

                    # 이동 시간 기록
                    travel_times.append(env.cumulative_travel_times.clone())

                    state = next_state

                # 에피소드 종료 후 보상 계산
                reward = compute_episode_reward(env, reward_type, alpha, beta, gamma, use_2opt=use_2opt)
                reward = clip_rewards(reward)

                # 보상에 NaN이 있는지 확인
                if torch.isnan(reward).any():
                    print(f"NaN detected in rewards at Epoch {epoch}, Batch {batch_idx+1}")
                    reward = torch.tensor(0.0, device=device)

                # 할인율을 적용한 누적 보상 계산 (단일 보상)
                R = reward.item()
                returns = [R] * len(log_probs)
                returns = torch.tensor(returns, device=device)

                # 보상 표준화 (선택 사항)
                if returns.std() != 0:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-5)

                # 정책 손실과 가치 손실 계산
                policy_loss = []
                value_loss = []
                for log_prob, value, R in zip(log_probs, values, returns):
                    advantage = R - value
                    policy_loss.append(-log_prob * advantage)
                    value_loss.append(F.mse_loss(value, torch.tensor(R, device=device)))

                if policy_loss and value_loss:
                    policy_loss_total = torch.stack(policy_loss).mean()

                    # 엔트로피 보너스 추가
                    entropy_total = torch.stack(entropy_list).mean()
                    policy_loss_total = policy_loss_total - entropy_coeff * entropy_total

                    value_loss_total = torch.stack(value_loss).mean()
                    loss = policy_loss_total + value_loss_total
                    
                    # 역전파
                    optimizer_actor.zero_grad()
                    optimizer_critic.zero_grad()
                    loss.backward()
                    
                    # 그래디언트 클리핑
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                    
                    # 옵티마이저 스텝
                    optimizer_actor.step()
                    optimizer_critic.step()
                else:
                    loss = torch.tensor(0.0, device=device)

                # 보상과 이동 시간 로깅
                average_travel_time = torch.stack(travel_times).mean().item() if travel_times else 0.0

                # tqdm 진행 표시줄에 정보 업데이트
                epoch_pbar.set_description(f"에폭 {epoch}/{num_epochs} | 배치 {batch_idx+1}/{batch_size} | 보상 {reward.item():.2f} | 손실 {loss.item():.4f} | Temperature {temperature:.4f}")

                uav_logs = {}
                for i in range(env.num_uavs):
                    uav_logs[f"uav_{i}_travel_time"] = env.cumulative_travel_times[i].item()
                    uav_logs[f"uav_{i}_assignments"] = len(env.paths[i])

                # WandB에 로그 기록
                wandb.log({
                    "episode": episode,
                    "epoch": epoch,
                    "batch": batch_idx,
                    "policy_loss": policy_loss_total.item() if policy_loss else 0,
                    "value_loss": value_loss_total.item() if value_loss else 0,
                    "loss": loss.item(),
                    "reward": reward.item(),
                    "temperature": temperature,
                    "entropy": entropy_total.item() if policy_loss and value_loss else 0,
                    "actor_dropout": policy_net.module.actor_shared[-1].p if isinstance(policy_net, nn.DataParallel) else policy_net.actor_shared[-1].p,
                    "critic_dropout": policy_net.module.critic_shared[-1].p if isinstance(policy_net, nn.DataParallel) else policy_net.critic_shared[-1].p,
                    "actor_layers": len(policy_net.module.actor_shared) // 3 if isinstance(policy_net, nn.DataParallel) else len(policy_net.actor_shared) // 3,
                    "critic_layers": len(policy_net.module.critic_shared) // 3 if isinstance(policy_net, nn.DataParallel) else len(policy_net.critic_shared) // 3,
                    "gnn_hidden_dim": policy_net.module.gnn_encoder.layers[0].out_channels // policy_net.module.gnn_encoder.layers[0].heads if isinstance(policy_net, nn.DataParallel) else policy_net.gnn_encoder.layers[0].out_channels // policy_net.gnn_encoder.layers[0].heads,
                    "actor_hidden_dim": policy_net.module.actor_shared[0].in_features if isinstance(policy_net, nn.DataParallel) else policy_net.actor_shared[0].in_features,
                    "critic_hidden_dim": policy_net.module.critic_shared[0].in_features if isinstance(policy_net, nn.DataParallel) else policy_net.critic_shared[0].in_features,
                    **uav_logs,  # 각 UAV별 이동 시간 및 할당 미션 수를 포함
                    "entropy_coeff": entropy_coeff,  # 엔트로피 가중치 로깅
                    "action_probs": wandb.Histogram(F.softmax(action_logits.detach().cpu(), dim=-1).numpy())
                })

                episode += 1

            # 에폭 별로 온도 업데이트
            if temperature > temperature_min:
                temperature *= temperature_decay
                temperature = max(temperature, temperature_min)
            
            # 학습률 스케줄러 업데이트
            scheduler_actor.step()
            scheduler_critic.step()

            # 검증
            if epoch % 1 == 0:
                validation_reward = validate_model(val_env, policy_net, device, edge_index, batch, checkpoints_path, results_path, epoch, reward_type, alpha, beta, gamma, wandb_name, use_2opt=use_2opt)
                
                # 조기 종료 체크
                if validation_reward > best_validation_reward:
                    best_validation_reward = validation_reward
                    epochs_no_improve = 0
                    # 최적의 모델을 별도로 저장
                    best_model_path = os.path.join(checkpoints_path, f"best_model_epoch_{epoch}.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': policy_net.state_dict(),
                        'optimizer_actor_state_dict': optimizer_actor.state_dict(),
                        'optimizer_critic_state_dict': optimizer_critic.state_dict(),
                        'temperature': temperature
                    }, best_model_path)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"조기 종료가 {patience} 에폭 동안 개선되지 않아 트리거되었습니다.")
                        return

    except KeyboardInterrupt:
        print("학습이 중단되었습니다. 체크포인트를 저장합니다...")
        last_checkpoint_path = os.path.join(checkpoints_path, f"interrupted_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_actor_state_dict': optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': optimizer_critic.state_dict(),
            'temperature': temperature
        }, last_checkpoint_path)
        print(f"체크포인트가 저장되었습니다: {last_checkpoint_path}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    last_checkpoint_path = os.path.join(checkpoints_path, f"final_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': policy_net.state_dict(),
        'optimizer_actor_state_dict': optimizer_actor.state_dict(),
        'optimizer_critic_state_dict': optimizer_critic.state_dict(),
        'temperature': temperature
    }, last_checkpoint_path)
    print(f"학습이 완료되었습니다. 마지막 체크포인트가 저장되었습니다: {last_checkpoint_path}")

    wandb.finish()

# ============================
# 검증 및 테스트 함수
# ============================

def validate_model(env, policy_net, device, edge_index, batch, checkpoints_path, results_path, epoch, reward_type, alpha, beta, gamma, wandb_name="run", use_2opt=False):
    """
    정책 네트워크를 검증합니다.
    
    Args:
        env (MissionEnvironment): 검증 환경.
        policy_net (nn.Module): 정책 네트워크.
        device (torch.device): 실행 장치.
        edge_index (torch.Tensor): 엣지 인덱스.
        batch (torch.Tensor): 배치 인덱스.
        checkpoints_path (str): 검증 체크포인트 저장 경로.
        results_path (str): 가시화 저장 경로.
        epoch (int): 현재 에폭.
        reward_type (str): 보상 함수 유형.
        alpha (float): 혼합 보상 시 최대 소요 시간 패널티 가중치.
        beta (float): 혼합 보상 시 전체 소요 시간 합 패널티 가중치.
        gamma (float): travel_time_variance  패널티 가중치.
        wandb_name (str): WandB run 이름.
        
    Returns:
        float: 총 검증 보상.
    """
    policy_net.eval()  # 평가 모드로 전환
    state = env.reset()
    done = False
    total_reward = 0
    cumulative_travel_times = torch.zeros(env.num_uavs, device=device)
    paths = [[] for _ in range(env.num_uavs)]

    with torch.no_grad():  # 평가 시에는 기울기 계산 비활성화
        while not done:
            positions = state['positions']
            uavs_info = positions.to(device)
            # 임무 완료 여부에 따라 액션 마스크 생성
            action_mask = create_action_mask(state, done=done)
            
            # 비용 행렬과 도착 시간 계산
            timetogo_matrix, dist_matrix = calculate_cost_matrix(positions, env.missions, env.speeds)

            # 정책 네트워크 순전파
            action_logits, _ = policy_net(
                env.missions, 
                edge_index, 
                batch, 
                uavs_info, 
                action_mask, 
                env.speeds,
                dist_matrix,
                timetogo_matrix,
            )
            
            # NaN 검사 추가
            if torch.isnan(action_logits).any():
                print(f"NaN detected in action_logits during validation at Epoch {epoch}")
            
            # UAV 선택 순서 결정 (준비된 UAV들만 포함)
            uav_order = compute_uav_order(env)
            # 탐험 없이 액터 정책에 따라 액션 선택 (온도= 낮음, 예: 매우 낮은 온도)
            actions = choose_action(action_logits, dist_matrix, timetogo_matrix, temperature=0.0001, uav_order=uav_order, global_action_mask=action_mask)

            # 환경 스텝
            next_state, travel_time, done = env.step(actions)

            # 보상 계산
            reward = compute_episode_reward(env, reward_type, alpha, beta, gamma, use_2opt=use_2opt)

            # 보상에 NaN이 있는지 확인
            if torch.isnan(torch.tensor(reward)).any():
                print(f"NaN detected in reward during validation at Epoch {epoch}")
                reward = 0.0

            total_reward += reward
            state = next_state

            for i in range(env.num_uavs):
                paths[i] = env.paths[i]
                cumulative_travel_times[i] = env.cumulative_travel_times[i]

    # 특성 중요도 계산
    timetogo_matrix, dist_matrix = calculate_cost_matrix(env.current_positions, env.missions, env.speeds)
    feature_importance = compute_feature_importance(
        policy_net, 
        env.missions, 
        edge_index, 
        batch, 
        env.current_positions,  # UAV 현재 위치를 입력으로 사용
        create_action_mask(env.get_state(), done=done),
        env.speeds,
        dist_matrix,
        timetogo_matrix,
        device
    )

    # 검증 모델 저장
    validation_model_save_path = os.path.join(checkpoints_path, f"validation_epoch_{epoch}.pth")
    torch.save({'epoch': epoch,
                        'model_state_dict': policy_net.state_dict()}, validation_model_save_path)
    
    # 가시화
    visualization_path = os.path.join(results_path, f"mission_paths_validation_epoch_{epoch}.png")
    visualize_results(
        env, 
        visualization_path,
        reward=total_reward,
        folder_name=f"Validation Epoch {epoch} - {wandb_name}"
    )
    
    # WandB에 로그 기록 (보상 및 특성 중요도)
    wandb.log({
        "validation_reward": total_reward,
        "validation_cumulative_travel_times": cumulative_travel_times.tolist(),
        "validation_mission_paths": wandb.Image(visualization_path),
        "epoch": epoch,
        "feature_importance": feature_importance,  # 특성 중요도 추가
        "actor_layers": len(policy_net.module.actor_shared) // 3 if isinstance(policy_net, nn.DataParallel) else len(policy_net.actor_shared) // 3,
        "critic_layers": len(policy_net.module.critic_shared) // 3 if isinstance(policy_net, nn.DataParallel) else len(policy_net.critic_shared) // 3,
        "actor_hidden_dim": policy_net.module.actor_shared[0].in_features if isinstance(policy_net, nn.DataParallel) else policy_net.actor_shared[0].in_features,
        "critic_hidden_dim": policy_net.module.critic_shared[0].in_features if isinstance(policy_net, nn.DataParallel) else policy_net.critic_shared[0].in_features,
        "action_probs": wandb.Histogram(F.softmax(action_logits.detach().cpu(), dim=-1).numpy())
        
    })

    policy_net.train()  # 다시 학습 모드로 전환
    
    return total_reward

def test_model(env, policy_net, device, edge_index, batch, checkpoint_path, results_path, reward_type, alpha, beta, gamma, wandb_name="run", use_2opt=False):
    """
    정책 네트워크를 테스트합니다.
    
    Args:
        env (MissionEnvironment): 테스트 환경.
        policy_net (nn.Module): 정책 네트워크.
        device (torch.device): 실행 장치.
        edge_index (torch.Tensor): 엣지 인덱스.
        batch (torch.Tensor): 배치 인덱스.
        checkpoint_path (str): 체크포인트 경로.
        results_path (str): 가시화 저장 경로.
        wandb_name (str): WandB run 이름.
    """

    # WandB 초기화
    wandb.init(project="multi_uav_mission", name=wandb_name)

    policy_net.eval()  # 평가 모드로 전환
    
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy_net.load_state_dict(checkpoint['model_state_dict'])
        if 'temperature' in checkpoint:
            temperature = checkpoint['temperature']
        print(f"체크포인트 '{checkpoint_path}'가 로드되었습니다. 테스트를 시작합니다.")
    else:
        print(f"체크포인트 '{checkpoint_path}'를 찾을 수 없습니다. 테스트를 종료합니다.")
        return

    state = env.reset()
    done = False
    total_reward = 0
    cumulative_travel_times = torch.zeros(env.num_uavs, device=device)
    paths = [[] for _ in range(env.num_uavs)]

    with torch.no_grad():  # 평가 시에는 기울기 계산 비활성화
        while not done:
            positions = state['positions']
            uavs_info = positions.to(device)
            # 임무 완료 여부에 따라 액션 마스크 생성
            action_mask = create_action_mask(state, done=done)
            
            # 비용 행렬과 도착 시간 계산
            timetogo_matrix, dist_matrix = calculate_cost_matrix(positions, env.missions, env.speeds)

            # 정책 네트워크 순전파
            action_logits, _ = policy_net(
                env.missions, 
                edge_index, 
                batch, 
                uavs_info, 
                action_mask, 
                env.speeds,
                dist_matrix,
                timetogo_matrix,
            )

            # UAV 선택 순서 결정 (준비된 UAV들만 포함)
            uav_order = compute_uav_order(env)
            # 탐험 없이 액터 정책에 따라 액션 선택 (온도= 낮음, 예: 매우 낮은 온도)
            actions = choose_action(action_logits, dist_matrix, timetogo_matrix, temperature=0.001, uav_order=uav_order, global_action_mask=action_mask)

            # 환경 스텝
            next_state, travel_time, done = env.step(actions)

            # 보상 계산
            reward = compute_episode_reward(env, reward_type, alpha, beta, gamma, use_2opt=use_2opt)


            total_reward += reward
            state = next_state

            for i in range(env.num_uavs):
                paths[i] = env.paths[i]
                cumulative_travel_times[i] = env.cumulative_travel_times[i]
                
    timetogo_matrix, dist_matrix = calculate_cost_matrix(env.current_positions, env.missions, env.speeds)
    
    # 특성 중요도 계산
    feature_importance = compute_feature_importance(
        policy_net, 
        env.missions, 
        edge_index, 
        batch, 
        env.current_positions,  # UAV 현재 위치를 입력으로 사용
        create_action_mask(env.get_state(), done=done),
        env.speeds,
        dist_matrix,
        timetogo_matrix,
        device
    )

    print(f"테스트 완료 - 총 보상: {total_reward}")
    visualization_path = os.path.join(results_path, "test_results.png")
    visualize_results(
        env, 
        visualization_path,
        reward=total_reward,
        folder_name=f"Test - {wandb_name}"
    )
    
    # WandB에 로그 기록 (보상 및 특성 중요도)
    wandb.log({
        "test_reward": total_reward,
        "test_cumulative_travel_times": cumulative_travel_times.tolist(),
        "test_mission_paths": wandb.Image(visualization_path),
        "feature_importance": feature_importance,  # 특성 중요도 추가
        "action_probs": wandb.Histogram(F.softmax(action_logits.detach().cpu(), dim=-1).numpy())
    })
    
    # # 선택 사항: 특성 중요도 시각화 이미지 저장 및 WandB에 업로드
    # feature_importance_path = os.path.join(results_path, "feature_importance_test.png")
    # visualize_feature_importance(feature_importance, feature_importance_path, title="Feature Importance - Test - " + wandb_name)
    # wandb.log({
    #     "feature_importance_test_bar": wandb.Bar(
    #         name="Feature Importance - Test",
    #         x=list(feature_importance.keys()),
    #         y=list(feature_importance.values())
    #     ),
    #     "feature_importance_test_image": wandb.Image(feature_importance_path)
    # })

    policy_net.train()  # 다시 학습 모드로 전환

# ============================
# 시각화 및 결과 저장 함수
# ============================

def visualize_results(env, save_path, reward=None, epsilon=None, policy_loss=None, value_loss=None, folder_name=None, 
                      num_epochs=None, num_uavs=None, num_missions=None, temperature=None):
    """
    미션 경로를 시각화하고 플롯을 저장합니다.
    
    Args:
        env (MissionEnvironment): 환경 인스턴스.
        save_path (str): 시각화 저장 경로.
        reward (float, optional): 표시할 보상.
        epsilon (float, optional): 표시할 epsilon 값.
        policy_loss (float, optional): 표시할 정책 손실.
        value_loss (float, optional): 표시할 가치 손실.
        folder_name (str, optional): 제목에 사용할 폴더 이름.
        num_epochs (int, optional): 학습 에포크 수.
        num_uavs (int, optional): UAV의 개수.
        num_missions (int, optional): 미션의 개수.
        temperature (float, optional): Boltzmann 탐험의 온도 매개변수.
    """
    plt.figure(figsize=(10, 10))
    missions = env.missions.cpu().numpy()
    plt.scatter(missions[:, 0], missions[:, 1], c='blue', marker='o', label='Missions')
    plt.scatter(missions[0, 0], missions[0, 1], c='green', marker='s', s=100, label='Start/End Point')

    # 노드 번호 및 시작점 텍스트 추가
    for i, (x, y) in enumerate(missions):
        if i == 0:
            plt.text(x, y, 'base', fontsize=12, color='green', fontweight='bold', ha='right', va='bottom')
        else:
            plt.text(x, y, f'{i}', fontsize=12, color='black', ha='right', va='bottom')

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, path in enumerate(env.paths):
        path_coords = missions[path]
        color = colors[i % len(colors)]
        plt.plot(path_coords[:, 0], path_coords[:, 1], marker='x', color=color, label=f'UAV {i} Path (Speed: {env.speeds[i].item():.2f})')

    plt.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    title = f'UAV MTSP - {folder_name}' if folder_name else 'UAV MTSP'
    plt.title(title)
    
    # Text annotations formatting
    annotations = []
    if reward is not None:
        annotations.append(f"Reward: {reward:.2f}")
    if temperature is not None:
        annotations.append(f"Temperature: {temperature:.4f}")
    if policy_loss is not None:
        annotations.append(f"Policy Loss: {policy_loss:.4f}")
    if value_loss is not None:
        annotations.append(f"Value Loss: {value_loss:.4f}")
    if num_epochs is not None:
        annotations.append(f"Epochs: {num_epochs}")
    if num_uavs is not None:
        annotations.append(f"UAVs: {num_uavs}")
    if num_missions is not None:
        annotations.append(f"Missions: {num_missions}")
    
    # Add the formatted text if there are any annotations
    if annotations:
        text = "\n".join(annotations)
        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_feature_importance(feature_importance, save_path, title="Feature Importance"):
    """
    특성 중요도를 시각화하고 플롯을 저장합니다.
    
    Args:
        feature_importance (dict): 각 특성의 중요도.
        save_path (str): 시각화 저장 경로.
        title (str): 그래프 제목.
    """
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())

    plt.figure(figsize=(10, 6))
    plt.bar(features, importances, color='skyblue')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ============================
# 시드 설정 함수
# ============================

def set_seed(seed):
    """
    재현성을 위해 랜덤 시드를 설정합니다.
    
    Args:
        seed (int): 시드 값.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ============================
# 메인 함수
# ============================

def main():
    """
    인자를 파싱하고 학습 또는 테스트를 시작하는 메인 함수.
    """
    parser = argparse.ArgumentParser(description="액터-크리틱 GNN을 이용한 다중 UAV 미션 할당 및 최적화")
    parser.add_argument('--config', type=str, default=None, help="Path to a json file with configuration parameters")
    parser.add_argument('--gpu', type=str, default='0', help="사용할 GPU 인덱스 (예: '0', '0,1', '0,1,2,3')")
    parser.add_argument('--num_uavs', type=int, default=3, help="UAV의 수")
    parser.add_argument('--num_missions', type=int, default=22, help="미션의 수")
    parser.add_argument('--embedding_dim', type=int, default=64, help="GNN 임베딩 차원")
    parser.add_argument('--gnn_hidden_dim', type=int, default=128, help="GNN 인코더의 숨겨진 차원")
    parser.add_argument('--actor_hidden_dim', type=int, default=128, help="액터 네트워크의 숨겨진 차원")
    parser.add_argument('--critic_hidden_dim', type=int, default=128, help="크리틱 네트워크의 숨겨진 차원")
    parser.add_argument('--actor_layers', type=int, default=8, help="액터 네트워크의 레이어 수")
    parser.add_argument('--critic_layers', type=int, default=8, help="크리틱 네트워크의 레이어 수")
    parser.add_argument('--num_layers', type=int, default=8, help="GNN 레이어 수")
    parser.add_argument('--heads', type=int, default=8, help="GNN Transformer 헤드 수")
    parser.add_argument('--num_epochs', type=int, default=500000, help="에폭 수")
    parser.add_argument('--batch_size', type=int, default=1024, help="배치 크기")
    # Remove epsilon related arguments
    # parser.add_argument('--epsilon_min', type=float, default=0.05, help="Epsilon 최소치")
    # parser.add_argument('--epsilon_decay', type=float, default=0.9999, help="Epsilon 감소율")
    parser.add_argument('--lr_actor', type=float, default=1e-4, help="액터 학습률")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="크리틱 학습률")
    parser.add_argument('--weight_decay_actor', type=float, default=1e-5, help="액터 옵티마이저의 weight decay")
    parser.add_argument('--weight_decay_critic', type=float, default=1e-5, help="크리틱 옵티마이저의 weight decay")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="기존 체크포인트의 경로")
    parser.add_argument('--test_mode', action='store_true', help="테스트 모드 활성화")
    parser.add_argument('--train_seed', type=int, default=632, help="Train 데이터셋 시드")
    parser.add_argument('--validation_seed', type=int, default=323, help="Validation 데이터셋 시드")
    parser.add_argument('--test_seed', type=int, default=743, help="Test 데이터셋 시드")
    parser.add_argument('--time_weight', type=float, default=2.0, help="보상 시간의 가중치")
    parser.add_argument('--lr_step_size', type=int, default=10000, help="학습률 스케줄러의 step size")
    parser.add_argument('--lr_gamma', type=float, default=0.01, help="학습률 스케줄러의 gamma 값")
    parser.add_argument('--entropy_coeff', type=float, default=0.1, help="정책 손실에 추가되는 엔트로피 가중치")
    
    # 드롭아웃 비율을 조정할 수 있는 인자 추가
    parser.add_argument('--gnn_dropout', type=float, default=0.3, help="GNN Transformer 인코더의 드롭아웃 비율")
    parser.add_argument('--actor_dropout', type=float, default=0.3, help="액터 네트워크의 드롭아웃 비율")
    parser.add_argument('--critic_dropout', type=float, default=0.3, help="크리틱 네트워크의 드롭아웃 비율")
    
    # 보상 함수 선택 인자 추가
    parser.add_argument('--reward_type', type=str, default='mixed', choices=['max', 'total', 'mixed'], help="보상 함수 유형: 'max', 'total', 'mixed'")
    parser.add_argument('--alpha', type=float, default=0.7, help="혼합 보상 시 최대 소요 시간 패널티 가중치 (reward_type='mixed'일 때 사용)")
    parser.add_argument('--beta', type=float, default=0.5, help="혼합 보상 시 전체 소요 시간 합 패널티 가중치 (reward_type='mixed'일 때 사용)")
    parser.add_argument('--gamma', type=float, default=0.3, help="travel_time_variance  패널티 가중치 (reward_type='mixed'일 때 사용)")
    
    # 2-opt 사용 여부 추가
    parser.add_argument('--use_2opt', action='store_true', help="훈련에서 2-opt 최적화 사용 여부")
    
    # 결과 디렉토리 추가
    parser.add_argument('--results_dir', type=str, default="/mnt/hdd2/attoman/GNN/results/boltzmann_prior/update/", help="결과 저장 디렉토리")
    
    # WandB 이름 인자 추가
    parser.add_argument('--name', type=str, default='boltzmann_prior', help="WandB run name")
    
    # Add temperature parameter for Boltzmann exploration
    parser.add_argument('--temperature', type=float, default=1.8, help="Boltzmann 탐험의 온도 매개변수")
    parser.add_argument('--temperature_decay', type=float, default=0.999999, help="Boltzmann 온도 감소율")
    parser.add_argument('--temperature_min', type=float, default=0.2, help="Boltzmann 온도의 최소값")
    
    args = parser.parse_args()
    
    # 먼저 --config 인자를 파싱
    args_config, remaining_argv = parser.parse_known_args()

    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_args = json.load(f)
            parser.set_defaults(**config_args)
            args = parser.parse_args()
        except Exception as e:
            print(f"설정 파일 로드 중 오류 발생: {e}")
            return

    # 나머지 인자 파싱 (커맨드라인 인자가 JSON 파일의 파라미터를 오버라이드)
    args = parser.parse_args()
    
    # 장치 설정
    # GPU 인자 처리
    num_gpus = 1  # 기본값 설정
    if torch.cuda.is_available() and args.gpu:
        gpu_indices = [int(x) for x in args.gpu.split(',')]
        num_gpus = len(gpu_indices)
        # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_indices))
        os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
        if num_gpus > 1:
            device = torch.device("cuda")
            print(f"{num_gpus}개의 GPU {gpu_indices}를 사용합니다.")
        else:
            device = torch.device(f"cuda:{gpu_indices[0]}")
            print(f"GPU {gpu_indices[0]}를 사용합니다.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS를 사용합니다.")
    else:
        device = torch.device("cpu")
        print("CPU를 사용합니다.")
    print(device)
    # 재현성을 위해 시드 설정
    set_seed(args.train_seed)

    # 데이터 생성
    train_data = MissionData(num_missions=args.num_missions, num_uavs=args.num_uavs, seed=args.train_seed, device=device)
    val_data = MissionData(num_missions=args.num_missions, num_uavs=args.num_uavs, seed=args.validation_seed, device=device)
    test_data = MissionData(num_missions=args.num_missions, num_uavs=args.num_uavs, seed=args.test_seed, device=device)

    # 환경 초기화 (2-opt 사용 여부 반영)
    train_env = MissionEnvironment(train_data.missions, train_data.uavs_start, train_data.uavs_speeds, device, mode='train', seed=args.train_seed, time_weight=args.time_weight, use_2opt=args.use_2opt)
    val_env = MissionEnvironment(val_data.missions, val_data.uavs_start, val_data.uavs_speeds, device, mode='val', seed=args.validation_seed, time_weight=args.time_weight, use_2opt=args.use_2opt)
    test_env = MissionEnvironment(test_data.missions, test_data.uavs_start, test_data.uavs_speeds, device, mode='test', seed=args.test_seed, time_weight=args.time_weight, use_2opt=args.use_2opt)

    # edge_index와 batch 생성
    edge_index = create_edge_index(args.num_missions, args.num_uavs).to(device)
    batch = torch.arange(args.num_uavs).repeat_interleave(args.num_missions).to(device)

    # 정책 네트워크 초기화
    policy_net = ImprovedActorCriticNetwork(
        num_missions=args.num_missions,
        num_uavs=args.num_uavs,
        embedding_dim=args.embedding_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        actor_hidden_dim=args.actor_hidden_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        gnn_dropout=args.gnn_dropout,
        actor_dropout=args.actor_dropout,
        critic_dropout=args.critic_dropout
    ).to(device)

    # DataParallel 사용 가능 시 적용
    if torch.cuda.is_available() and 'num_gpus' in locals() and num_gpus > 1:
        policy_net = nn.DataParallel(policy_net)

    # 옵티마이저 초기화 수정
    if isinstance(policy_net, nn.DataParallel):
        actor_params = list(policy_net.module.actor_shared.parameters()) + list(policy_net.module.actor_out.parameters()) + list(policy_net.module.actor_attention.parameters())
        critic_params = list(policy_net.module.critic_shared.parameters()) + list(policy_net.module.critic_out.parameters()) + list(policy_net.module.critic_attention.parameters())
    else:
        actor_params = list(policy_net.actor_shared.parameters()) + list(policy_net.actor_out.parameters()) + list(policy_net.actor_attention.parameters())
        critic_params = list(policy_net.critic_shared.parameters()) + list(policy_net.critic_out.parameters()) + list(policy_net.critic_attention.parameters())

    optimizer_actor = optim.Adam(actor_params, lr=args.lr_actor, weight_decay=args.weight_decay_actor)
    optimizer_critic = optim.Adam(critic_params, lr=args.lr_critic, weight_decay=args.weight_decay_critic)

    # 학습률 스케줄러 초기화
    scheduler_actor = optim.lr_scheduler.StepLR(optimizer_actor, step_size=args.lr_step_size, gamma=args.lr_gamma)
    scheduler_critic = optim.lr_scheduler.StepLR(optimizer_critic, step_size=args.lr_step_size, gamma=args.lr_gamma)


    # 결과 및 체크포인트 디렉토리 생성
    num_missions_folder = f"num_missions_{args.num_missions}"
    revision_folder = "revision"

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"현재 시간: {current_time}")

    # --name 인자를 사용하여 하위 폴더 생성
    name_folder = args.name
    base_dir = os.path.join(args.results_dir, num_missions_folder, current_time, name_folder, "update")


    # 학습 또는 테스트 모드 실행
    if args.test_mode:
        test_wandb_name = f"test_{args.name}"
        test_path = os.path.join(args.results_dir, f"num_missions_{args.num_missions}", "tests")
        os.makedirs(test_path, exist_ok=True)
        test_model(
            env=test_env, 
            policy_net=policy_net, 
            device=device, 
            edge_index=edge_index, 
            batch=batch, 
            checkpoint_path=args.checkpoint_path,
            results_path=test_path,
            reward_type=args.reward_type,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            wandb_name=test_wandb_name,
            use_2opt=args.use_2opt
        )
    else:
        images_path = os.path.join(base_dir, "images")
        checkpoints_path = os.path.join(base_dir, "checkpoints")
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(checkpoints_path, exist_ok=True)

        # 파라미터를 JSON 파일로 저장
        args_dict = vars(args)
        json_file_path = os.path.join(base_dir, 'config.json')
        with open(json_file_path, 'w') as f:
            json.dump(args_dict, f, indent=4)

        train_model(
            env=train_env, 
            val_env=val_env, 
            policy_net=policy_net, 
            optimizer_actor=optimizer_actor,
            optimizer_critic=optimizer_critic,
            scheduler_actor=scheduler_actor,
            scheduler_critic=scheduler_critic,
            num_epochs=args.num_epochs, 
            batch_size=args.batch_size, 
            device=device, 
            edge_index=edge_index, 
            batch=batch, 
            temperature=args.temperature,
            lr_gamma=args.lr_gamma,
            reward_type=args.reward_type,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            entropy_coeff=args.entropy_coeff,
            checkpoint_path=args.checkpoint_path,
            results_path=images_path,
            checkpoints_path=checkpoints_path,
            patience=5000,
            wandb_name=args.name,
            use_2opt=args.use_2opt
        )

if __name__ == "__main__":
    main()
