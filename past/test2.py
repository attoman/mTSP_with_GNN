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
import shap
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import seaborn as sns
import graphviz

# ============================
# 유틸리티 함수
# ============================

def calculate_distance(mission1, mission2):
    """두 미션 간의 유클리드 거리 계산"""
    return torch.sqrt(torch.sum((mission1 - mission2) ** 2))

def calculate_travel_time(distance, speed):
    """거리와 속도를 기반으로 이동 시간 계산"""
    return distance / (speed + 1e-5)  # 0으로 나누는 것을 방지

def create_edge_index(num_missions, num_uavs):
    """
    UAV와 미션 수에 따라 동적으로 엣지 생성

    Args:
        num_missions (int): 미션 수
        num_uavs (int): UAV 수

    Returns:
        torch.Tensor: edge_index 텐서
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
    현재 상태를 기반으로 각 UAV에 대한 액션 마스크 생성

    Args:
        state (dict): 현재 상태로 'visited', 'reserved', 'ready_for_next_action' 텐서를 포함
        done (bool): 모든 미션이 완료되었는지 여부

    Returns:
        torch.Tensor: (num_uavs, num_missions) 형태의 액션 마스크 텐서
    """
    visited = state['visited']  # (num_missions,)
    reserved = state['reserved']  # (num_missions,)
    ready = state['ready_for_next_action']  # (num_uavs,)

    num_uavs = ready.size(0)
    num_missions = visited.size(0)

    # 이미 방문했거나 예약된 미션을 마스킹
    action_mask = visited.unsqueeze(0).repeat(num_uavs, 1) | reserved.unsqueeze(0).repeat(num_uavs, 1)

    # 시작 지점(미션[0])을 모든 임무를 완료하기 전까지 마스킹
    action_mask[:, 0] = True

    # 모든 임무가 완료되면 시작 지점으로 이동 허용
    if visited[1:].all().item():
        action_mask[:, 0] = False

    # 준비되지 않은 UAV에 대해 모든 액션을 마스킹
    action_mask[~ready] = True  # 준비되지 않은 UAV는 액션을 취할 수 없음

    return action_mask  # (num_uavs, num_missions)


def calculate_cost_matrix(uav_positions, mission_coords, speeds):
    """
    거리 및 예상 소요 시간 행렬 계산

    Args:
        uav_positions (torch.Tensor): UAV 위치들. (num_uavs, 2)
        mission_coords (torch.Tensor): 미션 좌표들. (num_missions, 2)
        speeds (torch.Tensor): UAV 속도들. (num_uavs,)

    Returns:
        tuple: (timetogo_matrix, dist_matrix)
    """

    dist_matrix = torch.cdist(uav_positions, mission_coords)
    timetogo_matrix = dist_matrix / (speeds.unsqueeze(1) + 1e-5)
    
    # NaN을 0으로 대체
    dist_matrix = torch.nan_to_num(dist_matrix, nan=1e+6)
    timetogo_matrix = torch.nan_to_num(timetogo_matrix, nan=1e+6)

    return timetogo_matrix, dist_matrix

def get_policy_module(policy_net):
    """
    DataParallel 적용 시 실제 모듈 가져오기

    Args:
        policy_net (nn.Module): 정책 네트워크

    Returns:
        nn.Module: 실제 정책 모듈
    """
    return policy_net.module if isinstance(policy_net, nn.DataParallel) else policy_net

def set_seed(seed):
    """
    재현성을 위한 랜덤 시드 설정

    Args:
        seed (int): 시드 값
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_uav_order(env):
    """
    UAV들이 액션을 선택하는 순서 결정

    Args:
        env (MissionEnvironment): 환경 인스턴스

    Returns:
        list: UAV 인덱스의 순서가 지정된 리스트
    """
    # 누적 이동 시간을 기준으로 UAV 정렬 (짧은 시간 우선)
    cumulative_travel_times = env.cumulative_travel_times.cpu().numpy()
    uav_indices = list(range(env.num_uavs))
    uav_order = sorted(uav_indices, key=lambda i: cumulative_travel_times[i])
    return uav_order

# ============================
# 보상 함수
# ============================

def compute_reward_max_time(env, max_possible_time=1000, use_3opt=True):
    """
    최대 이동 시간을 기반으로 보상 계산, 선택적으로 3-opt 최적화 적용

    Args:
        env (MissionEnvironment): 환경 인스턴스
        max_possible_time (float): 스케일링을 위한 최대 가능 시간
        use_3opt (bool): 3-opt 최적화를 적용할지 여부

    Returns:
        float: 보상 값
    """
    try:
        if use_3opt:
            optimized_paths = [apply_3opt(path, env.missions) for path in env.paths]
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

def compute_reward_total_time(env, max_possible_time=1000, use_3opt=True):
    """
    총 이동 시간을 기반으로 보상 계산, 선택적으로 3-opt 최적화 적용

    Args:
        env (MissionEnvironment): 환경 인스턴스
        max_possible_time (float): 스케일링을 위한 최대 가능 시간
        use_3opt (bool): 3-opt 최적화를 적용할지 여부

    Returns:
        float: 보상 값
    """
    try:
        if use_3opt:
            optimized_paths = [apply_3opt(path, env.missions) for path in env.paths]
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

def compute_reward_mixed(env, alpha=0.5, beta=0.3, gamma=0.1, max_possible_time=1000, use_3opt=True):
    """
    최대, 총, 이동 시간 분산을 혼합하여 보상 계산, 선택적으로 3-opt 적용

    Args:
        env (MissionEnvironment): 환경 인스턴스
        alpha (float): 최대 이동 시간 가중치
        beta (float): 총 이동 시간 가중치
        gamma (float): 이동 시간 분산 가중치
        max_possible_time (float): 스케일링을 위한 최대 가능 시간
        use_3opt (bool): 3-opt 최적화를 적용할지 여부

    Returns:
        float: 보상 값
    """
    try:
        if use_3opt:
            optimized_paths = [apply_3opt(path, env.missions) for path in env.paths]
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
        workload_distribution = 1 / (mission_distribution.var() + 1e-8)
        
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


def compute_episode_reward(env, reward_type='total', alpha=0.5, beta=0.5, gamma=0.5, max_possible_time=1000, use_3opt=True):
    """
    지정된 보상 유형에 따라 에피소드 보상 계산

    Args:
        env (MissionEnvironment): 환경 인스턴스
        reward_type (str): 보상 유형 ('max', 'total', 'mixed')
        alpha (float): 최대 이동 시간 가중치 ('mixed'에서 사용)
        beta (float): 총 이동 시간 가중치 ('mixed'에서 사용)
        gamma (float): 이동 시간 분산 가중치 ('mixed'에서 사용)
        max_possible_time (float): 스케일링을 위한 최대 가능 시간
        use_3opt (bool): 3-opt 최적화를 적용할지 여부

    Returns:
        float: 계산된 보상
    """
    if reward_type == 'max':
        return compute_reward_max_time(env, max_possible_time, use_3opt)
    elif reward_type == 'total':
        return compute_reward_total_time(env, max_possible_time, use_3opt)
    elif reward_type == 'mixed':
        return compute_reward_mixed(env, alpha, beta, gamma, max_possible_time, use_3opt)
    else:
        raise ValueError(f"알 수 없는 reward_type: {reward_type}")

def clip_rewards(rewards, min_value=-1000, max_value=1000):
    """
    훈련 안정화를 위해 보상을 지정된 범위로 클리핑

    Args:
        rewards (float or list): 보상 값들
        min_value (float): 최소 보상 값
        max_value (float): 최대 보상 값

    Returns:
        torch.Tensor: 클리핑된 보상
    """
    rewards = torch.tensor(rewards, dtype=torch.float32)
    return torch.clamp(rewards, min=min_value, max=max_value)

# ============================
# 기능 중요도 함수
# ============================

def compute_feature_importance(policy_net, mission_coords, edge_index, batch, uavs_info, action_mask, speeds, dist_matrix, timetogo_matrix, device):
    """
    그레이디언트 기반 기능 중요도 계산

    Args:
        policy_net (nn.Module): 정책 네트워크
        mission_coords (torch.Tensor): 미션 좌표들. (num_missions, 2)
        edge_index (torch.Tensor): GNN용 엣지 인덱스
        batch (torch.Tensor): 배치 인덱스. (num_nodes,)
        uavs_info (torch.Tensor): UAV 위치 정보. (num_uavs, 2)
        action_mask (torch.Tensor): 액션 마스크. (num_uavs, num_missions)
        speeds (torch.Tensor): UAV 속도들. (num_uavs,)
        dist_matrix (torch.Tensor): 거리 행렬. (num_uavs, num_missions)
        timetogo_matrix (torch.Tensor): 예상 소요 시간 행렬. (num_uavs, num_missions)
        device (torch.device): 연산에 사용할 디바이스

    Returns:
        dict: 기능 중요도 점수
    """
    policy_net.eval()
    policy_net.zero_grad()

    try:
        model = get_policy_module(policy_net)

        # 그레이디언트 계산 활성화
        mission_coords = mission_coords.clone().detach().requires_grad_(True).to(device)
        uavs_info = uavs_info.clone().detach().requires_grad_(True).to(device)
        speeds = speeds.clone().detach().requires_grad_(True).to(device)
        dist_matrix = dist_matrix.clone().detach().requires_grad_(True).to(device)
        timetogo_matrix = timetogo_matrix.clone().detach().requires_grad_(True).to(device)

        # 순전파
        action_logits, state_values = model(
            mission_coords,
            edge_index,
            batch,
            uavs_info,
            action_mask,
            speeds,
            dist_matrix,
            timetogo_matrix
        )

        # 상태 값의 평균 계산
        state_values_mean = state_values.mean()

        # 역전파
        state_values_mean.backward()

        # 그레이디언트 추출
        feature_gradients = {
            "mission_x": mission_coords.grad[:, 0].abs().mean().item(),
            "mission_y": mission_coords.grad[:, 1].abs().mean().item(),
            "uav_x": uavs_info.grad[:, 0].abs().mean().item(),
            "uav_y": uavs_info.grad[:, 1].abs().mean().item(),
            "speed": speeds.grad.abs().mean().item(),
            "distance": dist_matrix.grad.abs().mean().item(),
            "time_to_go": timetogo_matrix.grad.abs().mean().item()
        }

        return feature_gradients
    except Exception as e:
        print(f"compute_feature_importance에서 오류 발생: {e}")
        return {}

# ============================
# 데이터 클래스
# ============================

class MissionData:
    """
    미션 데이터를 생성하고 관리하는 클래스
    """
    def __init__(self, num_missions=20, num_uavs=3, seed=None, device='cpu'):
        self.num_missions = num_missions
        self.num_uavs = num_uavs
        self.seed = seed
        self.device = device
        self.missions, self.uavs_start, self.uavs_speeds = self.generate_data()

    def generate_data(self):
        """랜덤한 미션 좌표, UAV 시작 위치, 속도 생성"""
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
        uavs_speeds = torch.randint(5, 30, (self.num_uavs,), dtype=torch.float)
        return missions.to(self.device), uavs_start.to(self.device), uavs_speeds.to(self.device)

    def reset_data(self, seed=None):
        """새로운 시드로 미션 데이터 재설정"""
        self.seed = seed
        self.missions, self.uavs_start, self.uavs_speeds = self.generate_data()

# ============================
# 강화 학습 환경
# ============================

class MissionEnvironment:
    """
    다중 UAV 미션 할당 강화 학습 환경
    """
    def __init__(self, missions=None, uavs_start=None, uavs_speeds=None, device='cpu', mode='train', seed=None, time_weight=2.0, use_3opt=False):
        self.device = device
        self.mode = mode
        self.seed = seed
        self.num_missions = missions.size(0) if missions is not None else 20
        self.num_uavs = uavs_start.size(0) if uavs_start is not None else 3

        self.missions = missions
        self.uavs_start = uavs_start
        self.speeds = uavs_speeds
        self.use_3opt = use_3opt
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
        액션 실행 및 환경 상태 업데이트

        Args:
            actions (list): UAV들이 선택한 액션들

        Returns:
            tuple: (next_state, travel_times, done)
        """
        try:
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

            # 모든 임무(시작 지점 제외)가 완료되었는지 확인
            done = self.visited[1:].all()

            # 모든 임무가 완료되었으면 UAV들이 시작 지점으로 돌아가도록 설정
            if done:
                for i in range(self.num_uavs):
                    if not torch.allclose(self.current_positions[i], self.missions[0]):
                        self.targets[i] = 0
                        self.ready_for_next_action[i] = False
                        self.remaining_distances[i] = calculate_distance(self.current_positions[i], self.missions[0])

            # 모든 UAV가 시작 지점에 돌아왔는지 확인
            fully_done = done and all([
                self.ready_for_next_action[i] and torch.allclose(self.current_positions[i], self.missions[0])
                for i in range(self.num_uavs)
            ])

            return self.get_state(), travel_times, fully_done  # 모든 UAV가 돌아오면 종료
        except Exception as e:
            print(f"환경 step에서 오류 발생: {e}")
            return self.get_state(), travel_times, False


# ============================
# GNN Transformer 인코더
# ============================

class EnhancedGNNTransformerEncoder(nn.Module):
    """
    잔차 연결이 있는 GNN Transformer 인코더
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=8, heads=8, dropout=0.3):
        super(EnhancedGNNTransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.hidden_channels = hidden_channels

        # 시간 특징 처리를 위한 인코더 추가
        self.time_encoder = nn.Sequential(
            nn.Linear(3, hidden_channels),  # 누적시간, 예상도착시간, 시간효율성
            nn.ReLU(),
            nn.LayerNorm(hidden_channels)
        )
        
        # 공간 특징 처리를 위한 인코더 추가
        self.spatial_encoder = nn.Sequential(
            nn.Linear(2, hidden_channels),  # 좌표 특징
            nn.ReLU(),
            nn.LayerNorm(hidden_channels)
        )
        
        # GNN 레이어
        self.layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        for layer in range(num_layers):
            conv = TransformerConv(
                in_channels if layer == 0 else hidden_channels * heads,
                hidden_channels,
                heads=heads,
                dropout=dropout
            )
            self.layers.append(conv)
            
            # Residual 프로젝션
            if layer == 0 and in_channels != hidden_channels * heads:
                projection = nn.Linear(in_channels, hidden_channels * heads)
            else:
                projection = nn.Identity()
            self.residual_projections.append(projection)
        
        # 특징 결합 레이어
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout)
        )
        
        # 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x, edge_index, batch):
        """
        Args:
            x (torch.Tensor): 입력 특징 (num_nodes, in_channels)
            edge_index (torch.Tensor): 엣지 인덱스
            batch (torch.Tensor): 배치 인덱스
            time_features (torch.Tensor): 시간 관련 특징 (num_nodes, 3)
        """
        # 시간 관련 특징 추출 (x 텐서에서)
        time_features = torch.cat([
            x[:, -3:],  # timetogo, cumulative_time, time_efficiency
        ], dim=-1)
        
        # 시간 특징 처리
        time_encoded = self.time_encoder(time_features)  # (num_nodes, hidden_channels)
        
        # 공간 특징 처리 (x의 좌표 부분만 사용)
        spatial_features = x[:, :2]  # 좌표 특징 추출
        spatial_encoded = self.spatial_encoder(spatial_features)  # (num_nodes, hidden_channels)
        
        # GNN 레이어 처리
        current_features = x
        for layer, conv in enumerate(self.layers):
            # Residual 연결
            residual = self.residual_projections[layer](current_features)
            
            # TransformerConv 레이어
            conv_out = conv(current_features, edge_index)
            conv_out = F.relu(conv_out)
            conv_out = F.dropout(conv_out, p=0.1, training=self.training)
            
            # Residual 더하기
            current_features = conv_out + residual
        
        # 특징 결합
        combined_features = torch.cat([
            current_features,
            time_encoded
        ], dim=-1)
        
        # 특징 융합
        fused_features = self.feature_fusion(combined_features)
        
        # 최종 출력
        output = self.output_layer(fused_features)
        
        return output

# ============================
# 액터-크리틱 네트워크
# ============================

class ImprovedActorCriticNetwork(nn.Module):
    def __init__(self, num_missions, num_uavs, embedding_dim=64, gnn_hidden_dim=64,
                 actor_hidden_dim=128, critic_hidden_dim=128,
                 num_layers=8, heads=8,
                 gnn_dropout=0.3, actor_dropout=0.3, critic_dropout=0.3):
        super(ImprovedActorCriticNetwork, self).__init__()
        
        # 공유 레이어
        self.actor_shared = nn.Sequential(
            nn.Linear(embedding_dim * 2, actor_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(actor_hidden_dim),
            nn.Dropout(actor_dropout)
        )
        
        # Actor 출력 레이어
        self.actor_out = nn.Linear(actor_hidden_dim, num_missions)
        
        # Actor Attention
        self.actor_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=heads,
            dropout=actor_dropout
        )
        
        # Critic 네트워크
        self.critic_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, critic_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(critic_hidden_dim),
            nn.Dropout(critic_dropout),
            nn.Linear(critic_hidden_dim, 1)
        )
        
        # GNN Transformer Encoder
        self.gnn_encoder = EnhancedGNNTransformerEncoder(
            in_channels=8,  # 좌표(2) + 마스크(1) + 속도(1) + 거리(1) + 시간(3)
            hidden_channels=gnn_hidden_dim,
            out_channels=embedding_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=gnn_dropout
        )
        
        # 시간 특징 인코더
        self.time_encoder = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(gnn_dropout)
        )
        
        # 특징 결합 레이어
        self.feature_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(gnn_dropout)
        )

    def forward(self, mission_coords, edge_index, batch, uavs_info, action_mask,
                speeds, dist_matrix, timetogo_matrix):
        # 시간 특징 준비
        time_features = torch.cat([
            timetogo_matrix.unsqueeze(-1),
            torch.zeros_like(timetogo_matrix.unsqueeze(-1)),  # 누적 시간
            torch.zeros_like(timetogo_matrix.unsqueeze(-1))   # 시간 효율성
        ], dim=-1)
        
        # 시간 특징 인코딩
        time_encoded = self.time_encoder(time_features)
        
        # 입력 특징 준비
        mask_embedded = action_mask.unsqueeze(-1).float()
        speeds_embedded = speeds.unsqueeze(-1).unsqueeze(1).repeat(1, mission_coords.size(0), 1)
        dist_embedded = dist_matrix.unsqueeze(-1)
        timetogo_embedded = timetogo_matrix.unsqueeze(-1)
        
        # 모든 특징 결합
        combined_features = torch.cat([
            mission_coords,
            mask_embedded,
            speeds_embedded,
            dist_embedded,
            timetogo_embedded,
            time_encoded
        ], dim=-1)
        
        # GNN 인코더 통과
        gnn_output = self.gnn_encoder(combined_features, edge_index, batch)
        
        # 시간 어텐션 적용
        time_context, _ = self.actor_attention(time_encoded, time_encoded, time_encoded)
        
        # 특징 결합
        fused_features = self.feature_fusion(torch.cat([
            gnn_output,
            time_encoded,
            time_context
        ], dim=-1))
        
        # Actor와 Critic 출력
        actor_features = self.actor_shared(fused_features)
        action_logits = self.actor_out(actor_features)
        state_values = self.critic_network(fused_features)
        
        return action_logits, state_values.squeeze(-1)

# ============================
# 액션 선택 함수
# ============================

def choose_action(action_logits, dist_matrix, timetogo_matrix, temperature, uav_order, global_action_mask=None):
    """
    가변 UAV 및 미션 수에 대응하는 유연한 액션 선택 함수.

    Args:
        action_logits (torch.Tensor): 액션에 대한 로짓. (num_uavs, num_missions)
        dist_matrix (torch.Tensor): 거리 행렬. (num_uavs, num_missions)
        temperature (float): 볼츠만 탐색 온도.
        uav_order (list): UAV 인덱스의 순서가 지정된 리스트.
        global_action_mask (torch.Tensor, optional): 여러 UAV가 동일한 미션을 선택하지 못하도록 하는 글로벌 액션 마스크.

    Returns:
        list: 각 UAV에 대한 선택된 액션.
    """
    num_uavs, num_missions = action_logits.shape
    actions = [-1] * num_uavs  # 액션 초기화

    for i in uav_order:
        if global_action_mask is not None:
            # UAV i에 대한 선택 가능한 액션 얻기
            available_actions = (global_action_mask[i] == 0).nonzero(as_tuple=True)[0].tolist()
            if not available_actions:
                continue  # 선택 가능한 액션이 없으면 스킵

            logits_i = action_logits[i, available_actions]
            distances = dist_matrix[i, available_actions]
            times = timetogo_matrix[i, available_actions]

            # 거리에 기반하여 로짓 조정
            """ mean_distance = distances.mean()
            std_distance = distances.std() + 1e-5  # 0으로 나누는 것을 방지
            z_scores = (distances - mean_distance) / std_distance
            weighted_logits = logits_i - z_scores / temperature """
            
            # 시간에 기반하여 로짓 조정
            mean_time = times.mean()
            std_time = times.std() + 1e-5  # 0으로 나누는 것을 방지
            z_scores = (times - mean_time) / std_time
            weighted_logits = logits_i - z_scores / temperature

            # 로짓 안정화
            weighted_logits = weighted_logits - weighted_logits.max()

            probs_i = F.softmax(weighted_logits, dim=-1).detach().cpu().numpy()

            # 유효하지 않은 확률 처리
            if np.isnan(probs_i).any() or not np.isfinite(probs_i).all():
                chosen_action = random.choice(available_actions)
            else:
                chosen_action = np.random.choice(available_actions, p=probs_i)

            # 다른 UAV가 동일한 미션을 선택하지 못하도록 글로벌 액션 마스크 업데이트
            for j in range(num_uavs):
                if j != i:
                    global_action_mask[j, chosen_action] = True
        else:
            logits_i = action_logits[i] / temperature - dist_matrix[i] / temperature
            logits_i = logits_i - logits_i.max()  # 로짓 안정화
            probs_i = F.softmax(logits_i, dim=-1).detach().cpu().numpy()
            if np.isnan(probs_i).any() or not np.isfinite(probs_i).all():
                chosen_action = random.randint(0, num_missions - 1)
            else:
                chosen_action = np.random.choice(num_missions, p=probs_i)

        actions[i] = chosen_action

    return actions


# ============================
# 3-opt 최적화 함수
# ============================

def apply_3opt(path, mission_coords):
    """
    UAV 경로를 최적화하기 위해 3-opt 알고리즘 적용

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
        for i in range(1, len(path) - 4):
            for j in range(i + 2, len(path) - 2):
                for k in range(j + 2, len(path)):
                    new_path = path[:i] + path[j:k] + path[i:j][::-1] + path[k:]
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
    각 UAV의 경로에 따라 총 이동 시간 계산

    Args:
        paths (list of lists): UAV 경로들
        mission_coords (torch.Tensor): 미션 좌표들. (num_missions, 2)
        speeds (torch.Tensor): UAV 속도들. (num_uavs,)

    Returns:
        torch.Tensor: 각 UAV의 총 이동 시간
    """
    total_travel_times = torch.zeros(len(paths), device=mission_coords.device)
    for i, path in enumerate(paths):
        travel_time = 0.0
        for j in range(len(path) - 1):
            distance = calculate_distance(mission_coords[path[j]], mission_coords[path[j + 1]]).item()
            travel_time += distance / (speeds[i].item() + 1e-5)  # 0으로 나누는 것을 방지
        total_travel_times[i] = travel_time
    return total_travel_times

# ============================
# 시각화 함수
# ============================

def visualize_results(env, save_path, reward=None, policy_loss=None, value_loss=None, folder_name=None, 
                      num_epochs=None, num_uavs=None, num_missions=None, temperature=None):
    """
    UAV 미션 경로를 시각화하고 그림을 저장합니다.

    Args:
        env (MissionEnvironment): 환경 인스턴스
        save_path (str): 시각화 저장 경로
        reward (float, optional): 보상 값
        policy_loss (float, optional): 정책 손실
        value_loss (float, optional): 가치 손실
        folder_name (str, optional): 그림 제목에 사용할 폴더 이름
        num_epochs (int, optional): 에폭 수
        num_uavs (int, optional): UAV 수
        num_missions (int, optional): 미션 수
        temperature (float, optional): 온도 파라미터
    """
    try:
        plt.figure(figsize=(10, 10))
        missions = env.missions.cpu().numpy()
        plt.scatter(missions[:, 0], missions[:, 1], c='blue', marker='o', label='Missions')
        plt.scatter(missions[0, 0], missions[0, 1], c='green', marker='s', s=100, label='Start/End Point')

        # 미션에 라벨 추가
        for i, (x, y) in enumerate(missions):
            label = 'base' if i == 0 else f'{i}'
            color = 'green' if i == 0 else 'black'
            plt.text(x, y, label, fontsize=12, color=color, ha='right', va='bottom')

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        for i, path in enumerate(env.paths):
            if len(path) < 2:
                continue  # 경로가 너무 짧으면 스킵
            path_coords = missions[path]
            color = colors[i % len(colors)]
            plt.plot(path_coords[:, 0], path_coords[:, 1], marker='x', color=color, label=f'UAV {i} Path (Speed: {env.speeds[i].item():.2f})')

        plt.legend()
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        title = f'UAV MTSP - {folder_name}' if folder_name else 'UAV MTSP'
        plt.title(title)
        
        # 텍스트 주석 추가
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
        
        if annotations:
            text = "\n".join(annotations)
            plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"visualize_results에서 오류 발생: {e}")

# ============================
# 훈련, 검증, 테스트 함수
# ============================

def train_model(env, val_env, policy_net, optimizer_actor, optimizer_critic, scheduler_actor, scheduler_critic,
                num_epochs, batch_size, device, edge_index, batch, temperature, lr_gamma, 
                reward_type='total', alpha=0.5, beta=0.5, gamma=0.5,
                entropy_coeff=0.01,  # 기본 값
                start_epoch=1, checkpoint_path=None, results_path=None, checkpoints_path=None, patience=10, wandb_name="run", use_3opt=False):
    """
    정책 네트워크를 훈련합니다.

    Args:
        env (MissionEnvironment): 훈련 환경
        val_env (MissionEnvironment): 검증 환경
        policy_net (nn.Module): 정책 네트워크
        optimizer_actor (torch.optim.Optimizer): 액터 옵티마이저
        optimizer_critic (torch.optim.Optimizer): 크리틱 옵티마이저
        scheduler_actor (torch.optim.lr_scheduler): 액터 학습률 스케줄러
        scheduler_critic (torch.optim.lr_scheduler): 크리틱 학습률 스케줄러
        num_epochs (int): 훈련 에폭 수
        batch_size (int): 배치 크기
        device (torch.device): 디바이스
        edge_index (torch.Tensor): 엣지 인덱스
        batch (torch.Tensor): 배치 인덱스
        temperature (float): 볼츠만 탐색 온도
        lr_gamma (float): 학습률 스케줄러 감마
        reward_type (str): 보상 유형
        alpha (float): 최대 이동 시간 가중치
        beta (float): 총 이동 시간 가중치
        gamma (float): 이동 시간 분산 가중치
        entropy_coeff (float): 엔트로피 계수
        start_epoch (int): 시작 에폭 번호
        checkpoint_path (str): 체크포인트 로드 경로
        results_path (str): 결과 저장 경로
        checkpoints_path (str): 체크포인트 저장 경로
        patience (int): 조기 종료를 위한 인내
        wandb_name (str): WandB 런 이름
        use_3opt (bool): 3-opt 최적화 사용 여부
    """
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
        "gnn_dropout": get_policy_module(policy_net).gnn_encoder.gnn_output[1].p,
        "actor_dropout": get_policy_module(policy_net).actor_shared[-1].p,
        "critic_dropout": get_policy_module(policy_net).critic_shared[-1].p,
        "num_missions": env.num_missions,
        "num_uavs": env.num_uavs,
        "reward_type": reward_type,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "entropy_coeff": entropy_coeff,
        "actor_layers": len(get_policy_module(policy_net).actor_shared) // 3,
        "critic_layers": len(get_policy_module(policy_net).critic_shared) // 3,
        "gnn_hidden_dim": get_policy_module(policy_net).gnn_encoder.layers[0].out_channels // get_policy_module(policy_net).gnn_encoder.layers[0].heads,
        "actor_hidden_dim": get_policy_module(policy_net).actor_shared[0].in_features,
        "critic_hidden_dim": get_policy_module(policy_net).critic_shared[0].in_features,
        "temperature": temperature
    })

    temperature_min = 0.5
    temperature_decay = 0.995

    best_validation_reward = -float('inf')
    epochs_no_improve = 0

    decision_tree_dataset = []  # 의사결정나무를 위한 데이터셋 초기화

    

    try:
        for epoch in tqdm(range(start_epoch, num_epochs + 1), desc="Epochs Progress"):
            epoch_pbar = tqdm(range(batch_size), desc=f"Epoch {epoch}/{num_epochs}", leave=False)
            for batch_idx in epoch_pbar:
                state = env.reset()
                done = False
                log_probs = []
                values = []
                entropy_list = []
                travel_times = []
                episode_data = []  # 상태-액션 쌍 수집

                while not done:
                    positions = state['positions']
                    uavs_info = positions.to(device)
                    action_mask = create_action_mask(state, done=done)
                    
                    timetogo_matrix, dist_matrix = calculate_cost_matrix(positions, env.missions, env.speeds)
                    
                    # 순전파
                    action_logits, state_values, actor_attn_weights, critic_attn_weights = get_policy_module(policy_net).forward(
                        env.missions, 
                        edge_index, 
                        batch, 
                        uavs_info, 
                        action_mask, 
                        env.speeds,
                        dist_matrix,
                        timetogo_matrix,
                        return_attn=True
                    )

                    if torch.isnan(action_logits).any():
                        action_logits = torch.where(torch.isnan(action_logits), 
                                                torch.zeros_like(action_logits), 
                                                action_logits)
                    
                    uav_order = compute_uav_order(env)

                    # 액션 선택
                    actions = choose_action(action_logits, dist_matrix, timetogo_matrix, temperature, uav_order, global_action_mask=action_mask)

                    # 로그 확률 및 상태 값 수집
                    for i, action in enumerate(actions):
                        if action != -1:
                            prob = F.softmax(action_logits[i], dim=-1)[action]
                            if torch.isnan(prob):
                                print(f"Epoch {epoch}, Batch {batch_idx+1}에서 action_probs[{i}, {action}]에 NaN이 발견되었습니다.")
                                prob = torch.tensor(1.0 / env.num_missions, device=device)
                            log_prob = torch.log(prob + 1e-10).squeeze()
                            log_probs.append(log_prob)
                            values.append(state_values[i].squeeze())

                    # 엔트로피 계산
                    entropy = -(F.softmax(action_logits, dim=-1) * torch.log(F.softmax(action_logits, dim=-1) + 1e-10)).sum(dim=-1).mean()
                    entropy_list.append(entropy)

                    # 환경 스텝
                    next_state, travel_time, done = env.step(actions)

                    # 이동 시간 기록
                    travel_times.append(env.cumulative_travel_times.clone())

                    # 의사결정나무를 위한 상태-액션 데이터 수집
                    for i, action in enumerate(actions):
                        if action != -1:
                            features = np.concatenate([
                                state['positions'].cpu().numpy().flatten(),
                                state['visited'].cpu().numpy().astype(int).flatten(),
                                state['reserved'].cpu().numpy().astype(int).flatten(),
                                env.speeds.cpu().numpy(),
                                state['remaining_distances'].cpu().numpy()
                            ])
                            episode_data.append(({
                                'positions': state['positions'].clone(),
                                'visited': state['visited'].clone(),
                                'reserved': state['reserved'].clone(),
                                'speeds': env.speeds.clone(),
                                'remaining_distances': state['remaining_distances'].clone()
                            }, action))
                    
                    state = next_state

                decision_tree_dataset.extend(episode_data)

                # 보상 계산
                reward = compute_episode_reward(env, reward_type, alpha, beta, gamma, use_3opt=use_3opt)
                reward = clip_rewards(reward)

                # 반환값 계산
                R = reward.item()
                returns = [R] * len(log_probs)
                returns = torch.tensor(returns, device=device)

                # 보상 정규화 (선택 사항)
                if returns.std() != 0:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-5)

                # 정책 및 가치 손실 계산
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

                # WandB에 로깅
                uav_logs = {}
                for i in range(env.num_uavs):
                    uav_logs[f"uav_{i}_travel_time"] = env.cumulative_travel_times[i].item()
                    uav_logs[f"uav_{i}_assignments"] = len(env.paths[i])

                wandb.log({
                    "episode": (epoch - 1) * batch_size + batch_idx + 1,
                    "epoch": epoch,
                    "batch": batch_idx,
                    "policy_loss": policy_loss_total.item() if policy_loss else 0,
                    "value_loss": value_loss_total.item() if value_loss else 0,
                    "loss": loss.item(),
                    "reward": reward.item(),
                    "temperature": temperature,
                    "entropy": entropy_total.item() if policy_loss and value_loss else 0,
                    **uav_logs,
                    "entropy_coeff": entropy_coeff,
                    "gradient_norm": torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0),
                    "learning_rate": optimizer_actor.param_groups[0]['lr'],
                    "action_entropy": entropy_total.item()
                })

            # 온도 업데이트
            temperature = temperature * (1 - epoch / num_epochs) + temperature_min

            # 검증
            if epoch % 1 == 0:
                validation_reward = validate_model(val_env, policy_net, device, edge_index, batch, checkpoints_path, results_path, epoch, reward_type, alpha, beta, gamma, wandb_name, use_3opt=use_3opt)
                
                # 조기 종료 체크
                if validation_reward > best_validation_reward:
                    best_validation_reward = validation_reward
                    epochs_no_improve = 0
                    # 최적의 모델 저장
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
                        print(f"{patience} 에폭 동안 개선되지 않아 조기 종료합니다.")
                        return

    except KeyboardInterrupt:
        print("훈련이 중단되었습니다. 체크포인트를 저장합니다...")
        interrupted_checkpoint = os.path.join(checkpoints_path, f"interrupted_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_actor_state_dict': optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': optimizer_critic.state_dict(),
            'temperature': temperature
        }, interrupted_checkpoint)
        print(f"체크포인트가 {interrupted_checkpoint}에 저장되었습니다.")
    except Exception as e:
        print(f"훈련 중 예기치 않은 오류 발생: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 최종 체크포인트 저장
    final_checkpoint = os.path.join(checkpoints_path, f"final_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': policy_net.state_dict(),
        'optimizer_actor_state_dict': optimizer_actor.state_dict(),
        'optimizer_critic_state_dict': optimizer_critic.state_dict(),
        'temperature': temperature
    }, final_checkpoint)
    print(f"훈련이 완료되었습니다. 최종 체크포인트가 {final_checkpoint}에 저장되었습니다.")

    # WandB 종료
    wandb.finish()

# ============================
# 검증 함수
# ============================

def validate_model(env, policy_net, device, edge_index, batch, checkpoints_path, results_path, epoch, reward_type, alpha, beta, gamma, wandb_name, use_3opt=False):
    """
    정책 네트워크를 검증합니다.

    Args:
        env (MissionEnvironment): 검증 환경.
        policy_net (nn.Module): 정책 네트워크.
        device (torch.device): 디바이스.
        edge_index (torch.Tensor): 엣지 인덱스.
        batch (torch.Tensor): 배치 인덱스.
        checkpoints_path (str): 검증 체크포인트를 저장할 경로.
        results_path (str): 검증 결과를 저장할 경로.
        epoch (int): 현재 에폭.
        reward_type (str): 보상 유형.
        alpha (float): 최대 이동 시간 가중치.
        beta (float): 총 이동 시간 가중치.
        gamma (float): 이동 시간 분산 가중치.
        wandb_name (str): WandB 런 이름.
        use_3opt (bool): 3-opt 최적화 사용 여부.

    Returns:
        float: 총 검증 보상.
    """
    policy_net.eval()
    state = env.reset()
    done = False
    total_reward = 0
    cumulative_travel_times = torch.zeros(env.num_uavs, device=device)
    paths = [[] for _ in range(env.num_uavs)]

    try:
        with torch.no_grad():
            while not done:
                positions = state['positions']
                uavs_info = positions.to(device)
                action_mask = create_action_mask(state, done=done)

                timetogo_matrix, dist_matrix = calculate_cost_matrix(positions, env.missions, env.speeds)

                # 순전파
                action_logits, state_values = get_policy_module(policy_net).forward(
                    env.missions,
                    edge_index,
                    batch,
                    uavs_info,
                    action_mask,
                    env.speeds,
                    dist_matrix,
                    timetogo_matrix,
                    return_attn=False
                )

                uav_order = compute_uav_order(env)
                actions = choose_action(action_logits, dist_matrix, timetogo_matrix, temperature=0.0001, uav_order=uav_order, global_action_mask=action_mask)

                next_state, travel_time, done = env.step(actions)
                reward = compute_episode_reward(env, reward_type, alpha, beta, gamma, use_3opt=use_3opt)
                total_reward += reward
                state = next_state

                for i in range(env.num_uavs):
                    paths[i] = env.paths[i]
                    cumulative_travel_times[i] = env.cumulative_travel_times[i]

        # 검증 모델 체크포인트 저장
        validation_model_save_path = os.path.join(checkpoints_path, f"validation_epoch_{epoch}.pth")
        torch.save({'epoch': epoch,
                    'model_state_dict': policy_net.state_dict()}, validation_model_save_path)

        # 결과 시각화
        visualization_path = os.path.join(results_path, f"mission_paths_validation_epoch_{epoch}.png")
        visualize_results(
            env,
            visualization_path,
            reward=total_reward,
            folder_name=f"Validation Epoch {epoch} - {wandb_name}"
        )

        # WandB에 검증 메트릭 로깅
        wandb.log({
            "validation_reward": total_reward,
            "validation_cumulative_travel_times": cumulative_travel_times.tolist(),
            "validation_mission_paths": wandb.Image(visualization_path),
        })
    except Exception as e:
        print(f"Epoch {epoch}의 검증 중 오류 발생: {e}")
    finally:
        policy_net.train()  # 훈련 모드로 전환

    return total_reward

# ============================
# 테스트 함수
# ============================

def test_model(env, policy_net, device, edge_index, batch, checkpoint_path, results_path, reward_type, alpha, beta, gamma, wandb_name="run", use_3opt=False):
    """
    정책 네트워크를 테스트합니다.

    Args:
        env (MissionEnvironment): 테스트 환경.
        policy_net (nn.Module): 정책 네트워크.
        device (torch.device): 디바이스.
        edge_index (torch.Tensor): 엣지 인덱스.
        batch (torch.Tensor): 배치 인덱스.
        checkpoint_path (str): 체크포인트 로드 경로.
        results_path (str): 테스트 결과를 저장할 경로.
        reward_type (str): 보상 유형.
        alpha (float): 최대 이동 시간 가중치.
        beta (float): 총 이동 시간 가중치.
        gamma (float): 이동 시간 분산 가중치.
        wandb_name (str): WandB 런 이름.
        use_3opt (bool): 3-opt 최적화 사용 여부.
    """
    # WandB 초기화
    wandb.init(project="multi_uav_mission", name=wandb_name)

    policy_net.eval()

    try:
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            policy_net.load_state_dict(checkpoint['model_state_dict'])
            temperature = checkpoint.get('temperature', 1.0)
            print(f"체크포인트 '{checkpoint_path}'를 로드했습니다. 테스트를 시작합니다.")
        else:
            print(f"체크포인트 '{checkpoint_path}'를 찾을 수 없습니다. 테스트를 종료합니다.")
            return

        state = env.reset()
        done = False
        total_reward = 0
        cumulative_travel_times = torch.zeros(env.num_uavs, device=device)
        paths = [[] for _ in range(env.num_uavs)]

        with torch.no_grad():
            while not done:
                positions = state['positions']
                uavs_info = positions.to(device)
                action_mask = create_action_mask(state, done=done)

                timetogo_matrix, dist_matrix = calculate_cost_matrix(positions, env.missions, env.speeds)

                # 순전파
                action_logits, state_values = get_policy_module(policy_net).forward(
                    env.missions,
                    edge_index,
                    batch,
                    uavs_info,
                    action_mask,
                    env.speeds,
                    dist_matrix,
                    timetogo_matrix,
                    return_attn=False
                )

                uav_order = compute_uav_order(env)
                actions = choose_action(action_logits, dist_matrix, timetogo_matrix, temperature=0.001, uav_order=uav_order, global_action_mask=action_mask)

                next_state, travel_time, done = env.step(actions)
                reward = compute_episode_reward(env, reward_type, alpha, beta, gamma, use_3opt=use_3opt)
                total_reward += reward
                state = next_state

                for i in range(env.num_uavs):
                    paths[i] = env.paths[i]
                    cumulative_travel_times[i] = env.cumulative_travel_times[i]

        print(f"테스트가 완료되었습니다. 총 보상: {total_reward}")
        visualization_path = os.path.join(results_path, "test_results.png")
        visualize_results(
            env,
            visualization_path,
            reward=total_reward,
            folder_name=f"Test - {wandb_name}"
        )

        # WandB에 테스트 메트릭 로깅
        wandb.log({
            "test_reward": total_reward,
            "test_cumulative_travel_times": cumulative_travel_times.tolist(),
            "test_mission_paths": wandb.Image(visualization_path),
        })

    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
    finally:
        policy_net.train()  # 훈련 모드로 전환

    wandb.finish()



# ============================
# 메인 함수
# ============================

def main():
    """
    인자를 파싱하고 훈련 또는 테스트를 시작합니다.
    """
    parser = argparse.ArgumentParser(description="다중 UAV 미션 할당 및 최적화를 위한 Actor-Critic GNN")
    parser.add_argument('--config', type=str, default=None, help="설정 파라미터가 포함된 JSON 파일 경로")
    parser.add_argument('--gpu', type=str, default='0', help="사용할 GPU 인덱스 (예: '0', '0,1')")
    parser.add_argument('--num_uavs', type=int, default=3, help="UAV 수")
    parser.add_argument('--num_missions', type=int, default=22, help="미션 수")
    parser.add_argument('--embedding_dim', type=int, default=64, help="GNN 임베딩 차원")
    parser.add_argument('--gnn_hidden_dim', type=int, default=64, help="GNN 인코더 히든 차원")
    parser.add_argument('--actor_hidden_dim', type=int, default=128, help="액터 네트워크 히든 차원")
    parser.add_argument('--critic_hidden_dim', type=int, default=128, help="크리틱 네트워크 히든 차원")
    parser.add_argument('--actor_layers', type=int, default=8, help="액터 레이어 수")
    parser.add_argument('--critic_layers', type=int, default=8, help="크리틱 레이어 수")
    parser.add_argument('--num_layers', type=int, default=8, help="GNN 레이어 수")
    parser.add_argument('--heads', type=int, default=8, help="GNN Transformer 헤드 수")
    parser.add_argument('--num_epochs', type=int, default=500000, help="훈련 에폭 수")
    parser.add_argument('--batch_size', type=int, default=1024, help="배치 크기")
    parser.add_argument('--lr_actor', type=float, default=1e-4, help="액터 학습률")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="크리틱 학습률")
    parser.add_argument('--weight_decay_actor', type=float, default=1e-5, help="액터 옵티마이저 가중치 감쇠")
    parser.add_argument('--weight_decay_critic', type=float, default=1e-5, help="크리틱 옵티마이저 가중치 감쇠")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="기존 체크포인트 로드 경로")
    parser.add_argument('--test_mode', action='store_true', help="테스트 모드 활성화")
    parser.add_argument('--train_seed', type=int, default=632, help="훈련 데이터셋 시드")
    parser.add_argument('--validation_seed', type=int, default=323, help="검증 데이터셋 시드")
    parser.add_argument('--test_seed', type=int, default=743, help="테스트 데이터셋 시드")
    parser.add_argument('--time_weight', type=float, default=2.0, help="보상에서 시간에 대한 가중치")
    parser.add_argument('--lr_step_size', type=int, default=10000, help="학습률 스케줄러 단계 크기")
    parser.add_argument('--lr_gamma', type=float, default=0.01, help="학습률 스케줄러 감마")
    parser.add_argument('--entropy_coeff', type=float, default=0.1, help="정책 손실에 대한 엔트로피 계수")
    parser.add_argument('--gnn_dropout', type=float, default=0.3, help="GNN Transformer 인코더 드롭아웃 비율")
    parser.add_argument('--actor_dropout', type=float, default=0.3, help="액터 네트워크 드롭아웃 비율")
    parser.add_argument('--critic_dropout', type=float, default=0.3, help="크리틱 네트워크 드롭아웃 비율")
    parser.add_argument('--reward_type', type=str, default='mixed', choices=['max', 'total', 'mixed'], help="보상 함수 유형")
    parser.add_argument('--alpha', type=float, default=0.7, help="최대 이동 시간 가중치 ('mixed'에서 사용)")
    parser.add_argument('--beta', type=float, default=0.5, help="총 이동 시간 가중치 ('mixed'에서 사용)")
    parser.add_argument('--gamma', type=float, default=0.3, help="이동 시간 분산 가중치 ('mixed'에서 사용)")
    parser.add_argument('--use_3opt', action='store_true', help="훈련에서 3-opt 최적화 사용 여부")
    parser.add_argument('--results_dir', type=str, default="/mnt/hdd2/attoman/GNN/results/test/", help="결과를 저장할 디렉토리")
    parser.add_argument('--name', type=str, default='boltzmann_prior', help="WandB 런 이름")
    parser.add_argument('--temperature', type=float, default=1.8, help="볼츠만 탐색을 위한 온도 파라미터")
    parser.add_argument('--temperature_decay', type=float, default=0.999999, help="볼츠만 온도 감쇠율")
    parser.add_argument('--temperature_min', type=float, default=0.2, help="볼츠만 탐색을 위한 최소 온도")

    args = parser.parse_args()

    # JSON 파일에서 설정 로드
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_args = json.load(f)
            parser.set_defaults(**config_args)
            args = parser.parse_args()
        except Exception as e:
            print(f"설정 파일 로드 중 오류 발생: {e}")
            return

    # 디바이스 설정
    num_gpus = 1
    if torch.cuda.is_available() and args.gpu:
        gpu_indices = [int(x) for x in args.gpu.split(',')]
        num_gpus = len(gpu_indices)
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_indices))
        device = torch.device("cuda" if num_gpus > 1 else f"cuda:{gpu_indices[0]}")
        print(f"{'여러 개의' if num_gpus > 1 else '단일'} GPU 사용: {args.gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS (Apple Silicon GPU) 사용")
    else:
        device = torch.device("cpu")
        print("CPU 사용")
    print(f"디바이스 설정: {device}")

    # 시드 설정
    set_seed(args.train_seed)

    # 데이터 생성
    train_data = MissionData(num_missions=args.num_missions, num_uavs=args.num_uavs, seed=args.train_seed, device=device)
    val_data = MissionData(num_missions=args.num_missions, num_uavs=args.num_uavs, seed=args.validation_seed, device=device)
    test_data = MissionData(num_missions=args.num_missions, num_uavs=args.num_uavs, seed=args.test_seed, device=device)

    # 환경 초기화
    train_env = MissionEnvironment(train_data.missions, train_data.uavs_start, train_data.uavs_speeds, device, mode='train', seed=args.train_seed, time_weight=args.time_weight, use_3opt=args.use_3opt)
    val_env = MissionEnvironment(val_data.missions, val_data.uavs_start, val_data.uavs_speeds, device, mode='val', seed=args.validation_seed, time_weight=args.time_weight, use_3opt=args.use_3opt)
    test_env = MissionEnvironment(test_data.missions, test_data.uavs_start, test_data.uavs_speeds, device, mode='test', seed=args.test_seed, time_weight=args.time_weight, use_3opt=args.use_3opt)

    # edge_index 및 batch 생성
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

    # DataParallel 적용
    if torch.cuda.is_available() and num_gpus > 1:
        policy_net = nn.DataParallel(policy_net)
        print("정책 네트워크에 DataParallel 적용.")

    # 옵티마이저 초기화
    model = get_policy_module(policy_net)
    actor_params = list(model.actor_shared.parameters()) + list(model.actor_out.parameters()) + list(model.actor_attention.parameters())
    critic_params = list(model.critic_shared.parameters()) + list(model.critic_out.parameters()) + list(model.critic_attention.parameters())

    optimizer_actor = optim.Adam(actor_params, lr=args.lr_actor, weight_decay=args.weight_decay_actor)
    optimizer_critic = optim.Adam(critic_params, lr=args.lr_critic, weight_decay=args.weight_decay_critic)

    # 학습률 스케줄러 초기화
    scheduler_actor = optim.lr_scheduler.ReduceLROnPlateau(optimizer_actor, mode='max', factor=0.1, patience=5000, verbose=True, min_lr=1e-6)
    scheduler_critic = optim.lr_scheduler.ReduceLROnPlateau(optimizer_critic, mode='max', factor=0.1, patience=5000, verbose=True, min_lr=1e-6)

    # 결과 및 체크포인트 디렉토리 생성
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(args.results_dir, f"num_missions_{args.num_missions}", current_time, args.name, "update")
    images_path = os.path.join(base_dir, "images")
    checkpoints_path = os.path.join(base_dir, "checkpoints")
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    # 설정을 JSON으로 저장
    args_dict = vars(args)
    config_file_path = os.path.join(base_dir, 'config.json')
    with open(config_file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)

    # 훈련 또는 테스트 실행
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
            use_3opt=args.use_3opt
        )
    else:
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
            use_3opt=args.use_3opt
        )

if __name__ == "__main__":
    main()
