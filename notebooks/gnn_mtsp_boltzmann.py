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
import torch.autograd as autograd
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
    각 UAV에 대해 모든 가능한 미션 경로를 연결하는 edge_index를 생성합니다.
    
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
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

def create_action_mask(state, done=False):
    """
    방문한 미션과 예약된 미션을 기반으로 액션 마스크를 생성합니다.
    임무가 완료된 경우에만 미션 0을 선택할 수 있도록 합니다.
    
    Args:
        state (dict): 현재 상태로 'visited'와 'reserved' 텐서를 포함합니다.
        done (bool): 모든 임무가 완료되었는지 여부.
        
    Returns:
        torch.Tensor: 액션 마스크 텐서.
    """
    visited = state['visited']
    reserved = state['reserved']
    action_mask = visited | reserved
    
    # 마지막 미션(시작/종료 지점)을 임무 중에는 방문할 수 없도록 설정
    action_mask[-1] = True
    
    # 모든 중간 미션이 완료되었을 때만 마지막 미션(시작/종료 지점)으로의 복귀를 허용
    if visited[1:-1].all():
        action_mask[-1] = False
    
    return action_mask

def calculate_cost_matrix(uav_positions, mission_coords, speeds):
    """
    거리와 UAV 속도를 기반으로 비용 행렬을 계산합니다.
    
    Args:
        uav_positions (torch.Tensor): UAV들의 현재 위치.
        mission_coords (torch.Tensor): 미션의 좌표.
        speeds (torch.Tensor): UAV들의 속도.
        
    Returns:
        torch.Tensor: 비용 행렬.
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
    
    Args:
        uav_positions (torch.Tensor): UAV들의 현재 위치.
        mission_coords (torch.Tensor): 미션의 좌표.
        speeds (torch.Tensor): UAV들의 속도.
        
    Returns:
        torch.Tensor: 도착 시간 텐서.
    """
    num_uavs = uav_positions.size(0)
    num_missions = mission_coords.size(0)
    
    arrival_times = torch.zeros((num_uavs, num_missions), device=uav_positions.device)
    
    for i in range(num_uavs):
        for j in range(num_missions):
            distance = calculate_distance(uav_positions[i], mission_coords[j])
            arrival_times[i, j] = distance / (speeds[i] + 1e-5)
    
    return arrival_times

# ============================
# 2-opt 알고리즘 구현
# ============================

def two_opt(route, mission_coords):
    """
    2-opt 알고리즘을 사용하여 경로를 최적화합니다.
    시작과 끝을 고정하여 0으로 유지합니다.
    
    Args:
        route (list): UAV의 현재 미션 경로.
        mission_coords (torch.Tensor): 모든 미션의 좌표.
        
    Returns:
        list: 최적화된 미션 경로.
    """
    best = route
    improved = True
    while improved:
        improved = False
        best_distance = calculate_total_distance(best, mission_coords)
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) -1):  # 마지막 지점은 고정
                if j - i == 1:
                    continue  # 연속된 미션은 스킵
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                new_distance = calculate_total_distance(new_route, mission_coords)
                if new_distance < best_distance:
                    best = new_route
                    best_distance = new_distance
                    improved = True
        if improved:
            continue
    return best

def calculate_total_distance(route, mission_coords):
    """
    주어진 경로의 총 거리를 계산합니다.
    
    Args:
        route (list): UAV의 미션 경로.
        mission_coords (torch.Tensor): 모든 미션의 좌표.
        
    Returns:
        float: 총 거리.
    """
    total = 0.0
    for i in range(len(route) - 1):
        total += calculate_distance(mission_coords[route[i]], mission_coords[route[i+1]]).item()
    return total

# ============================
# 보상 함수 구현
# ============================

def compute_reward_max_time(env):
    """
    최대 소요 시간을 최소화하는 보상 함수.
    
    Args:
        env (MissionEnvironment): 환경 인스턴스.
        
    Returns:
        float: 보상 값.
    """
    max_travel_time = env.cumulative_travel_times.max()
    reward = -max_travel_time  # 패널티로 적용
    return reward

def compute_reward_total_time(env):
    """
    전체 소요 시간 합을 최소화하는 보상 함수.
    
    Args:
        env (MissionEnvironment): 환경 인스턴스.
        
    Returns:
        float: 보상 값.
    """
    total_travel_time = env.cumulative_travel_times.sum()
    reward = -total_travel_time  # 패널티로 적용
    return reward

def compute_reward_mixed(env, alpha=0.5, beta=0.5):
    """
    최대 소요 시간과 전체 소요 시간 합을 모두 고려하는 혼합 보상 함수.
    
    Args:
        env (MissionEnvironment): 환경 인스턴스.
        alpha (float): 최대 소요 시간 패널티 가중치.
        beta (float): 전체 소요 시간 합 패널티 가중치.
        
    Returns:
        float: 보상 값.
    """
    max_travel_time = env.cumulative_travel_times.max()
    total_travel_time = env.cumulative_travel_times.sum()
    reward = -(alpha * max_travel_time + beta * total_travel_time)
    return reward


def choose_action(action_probs, temperature=1, uav_order=None, global_action_mask=None):
    """
    각 UAV에 대해 액션을 선택합니다. 볼츠만 탐험(temperature 기반)을 사용합니다.
    
    Args:
        action_probs (torch.Tensor): 정책 네트워크에서 나온 액션 확률.
        temperature (float): 소프트맥스 온도 매개변수. 낮을수록 결정적, 높을수록 랜덤.
        uav_order (list): UAV 선택 순서.
        global_action_mask (torch.Tensor): 글로벌 액션 마스크.
        
    Returns:
        list: 각 UAV에 대한 선택된 액션.
    """
    actions = []
    # 다른 UAV가 동일한 액션을 선택하지 않도록 로컬 액션 마스크 초기화
    local_action_mask = torch.zeros(action_probs.shape[1], dtype=torch.bool, device=action_probs.device)
    
    if uav_order is None:
        uav_order = list(range(action_probs.shape[0]))
    
    for i in uav_order:
        # 글로벌과 로컬 액션 마스크 결합
        if global_action_mask is not None:
            combined_mask = global_action_mask | local_action_mask
        else:
            combined_mask = local_action_mask
        masked_probs = action_probs[i].clone()
        masked_probs[combined_mask] = float('-inf')  # 선택되지 않도록 -inf으로 마스크
        masked_probs = F.softmax(masked_probs / temperature, dim=-1)  # 온도 매개변수 적용
        
        # 확률을 기반으로 액션 선택 (볼츠만 탐험)
        action = torch.multinomial(masked_probs, 1).item()  # 확률에 따라 액션 샘플링
        actions.append(action)
        local_action_mask[action] = True  # 다른 UAV가 동일한 액션을 선택하지 않도록 마스크
    
    return actions



def compute_uav_order(env):
    """
    예상 도착 시간을 기반으로 UAV 선택 순서를 결정합니다.
    
    Args:
        env (MissionEnvironment): 미션 환경.
        
    Returns:
        list: 정렬된 UAV 인덱스 리스트.
    """
    expected_arrival_times = []
    for i in range(env.num_uavs):
        if env.ready_for_next_action[i]:
            expected_arrival_times.append(0.0)
        else:
            if env.remaining_distances[i] == float('inf'):
                expected_arrival_times.append(float('inf'))
            else:
                expected_time = env.remaining_distances[i].item() / (env.speeds[i].item() + 1e-5)
                expected_arrival_times.append(expected_time)
    
    uav_indices = list(range(env.num_uavs))
    uav_order = sorted(uav_indices, key=lambda i: (expected_arrival_times[i], -env.speeds[i].item()))
    return uav_order

# ============================
# 특성 중요도 분석
# ============================

def compute_feature_importance(policy_net, mission_coords, edge_index, batch, uavs_info, action_mask, speeds, cost_matrix, arrival_times, device):
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
        cost_matrix (torch.Tensor): 비용 행렬. (num_uavs, num_missions)
        arrival_times (torch.Tensor): 도착 시간 텐서. (num_uavs, num_missions)
        device (torch.device): 실행 장치.
        
    Returns:
        dict: 각 특성의 중요도.
    """
    policy_net.zero_grad()
    
    # 입력 텐서에 기울기 계산을 위해 requires_grad 설정
    mission_coords = mission_coords.clone().detach().requires_grad_(True).to(device)  # (num_missions, 2)
    uavs_info = uavs_info.clone().detach().requires_grad_(True).to(device)          # (num_uavs, 2)
    speeds = speeds.clone().detach().requires_grad_(True).to(device)                # (num_uavs,)
    cost_matrix = cost_matrix.clone().detach().requires_grad_(True).to(device)      # (num_uavs, num_missions)
    arrival_times = arrival_times.clone().detach().requires_grad_(True).to(device)  # (num_uavs, num_missions)
    
    # 정책 네트워크 순전파
    action_probs, state_values = policy_net(
        mission_coords, 
        edge_index, 
        batch, 
        uavs_info, 
        action_mask, 
        speeds, 
        cost_matrix,
        arrival_times
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
    feature_gradients["cost"] = cost_matrix.grad.abs().mean().item()
    feature_gradients["arrival_time"] = arrival_times.grad.abs().mean().item()
    
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
        start_end_point = missions[0].clone()  # 첫 번째 미션을 시작/종료 지점으로 설정
        missions[-1] = start_end_point  # 마지막 미션을 시작/종료 지점과 동일하게 설정
        uavs_start = start_end_point.unsqueeze(0).repeat(self.num_uavs, 1)
        # uavs_speeds = torch.rand(self.num_uavs) * 9 * 3 + 1  # 속도는 1에서 30 사이
        uavs_speeds = torch.full((self.num_uavs,), 10.0)
        return missions.to(self.device), uavs_start.to(self.device), uavs_speeds.to(self.device)

    # 시드를 매번 다르게 설정하여 매번 다른 미션 데이터를 생성
    def reset_data(self, seed=None):
        """새로운 시드를 사용하여 미션 데이터를 재설정합니다."""
        if seed is None:
            self.seed = torch.randint(0, 10000, (1,)).item()
        else:
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
        """환경을 초기 상태로 리셋합니다."""
        self.current_positions = self.uavs_start.clone()
        self.visited = torch.zeros(self.num_missions, dtype=torch.bool, device=self.device)
        self.reserved = torch.zeros(self.num_missions, dtype=torch.bool, device=self.device)
        self.paths = [[] for _ in range(self.num_uavs)]
        self.cumulative_travel_times = torch.zeros(self.num_uavs, device=self.device)
        self.ready_for_next_action = torch.ones(self.num_uavs, dtype=torch.bool, device=self.device)
        self.targets = [-1] * self.num_uavs
        self.remaining_distances = torch.full((self.num_uavs,), float('inf'), device=self.device)

        self.visited[0] = True  # 시작 미션을 방문한 것으로 설정
        self.visited[-1] = False  # 마지막 미션(시작/종료 지점)은 아직 방문하지 않은 것으로 설정

        for i in range(self.num_uavs):
            self.paths[i].append(0)  # 각 UAV의 경로에 시작 미션 추가
        return self.get_state()
    
    def get_state(self):
        """현재 환경 상태를 반환합니다."""
        return {
            'positions': self.current_positions.clone(),
            'visited': self.visited.clone(),
            'reserved': self.reserved.clone(),
            'ready_for_next_action': self.ready_for_next_action.clone(),
            'remaining_distances': self.remaining_distances.clone(),
            'targets': self.targets
        }

    def step(self, actions):
        """
        액션을 실행하고 환경 상태를 업데이트합니다.
        
        Args:
            actions (list): UAV들이 선택한 액션.
        
        Returns:
            tuple: 다음 상태, 소요 시간 텐서, 종료 여부.
        """
        for i, action in enumerate(actions):
            if action is not None and self.ready_for_next_action[i] and not self.visited[action] and not self.reserved[action]:
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
    self-attention을 포함한 GNN Transformer 인코더.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, heads=8, dropout=0.3):
        super(EnhancedGNNTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
            in_channels = hidden_channels * heads

        self.gnn_output = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x, edge_index, batch):
        """
        GNN Transformer 인코더를 통한 순전파.
        
        Args:
            x (torch.Tensor): 노드 특성.
            edge_index (torch.Tensor): 엣지 인덱스.
            batch (torch.Tensor): 배치 인덱스.
        
        Returns:
            torch.Tensor: 노드 임베딩.
        """
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.gnn_output[1].p, training=self.training)

        x = self.gnn_output(x)

        return x

# ============================
# 액터-크리틱 네트워크
# ============================

class ImprovedActorCriticNetwork(nn.Module):
    """
    Enhanced GNN Transformer 인코더를 사용하는 액터-크리틱 네트워크.
    액터와 크리틱의 레이어 수와 숨겨진 차원을 동적으로 설정할 수 있습니다.
    """
    def __init__(self, num_missions, num_uavs, embedding_dim=64, gnn_hidden_dim=64, 
                 actor_hidden_dim=128, critic_hidden_dim=128,
                 actor_layers=3, critic_layers=3, num_layers=4, heads=8,
                 gnn_dropout=0.3, actor_dropout=0.3, critic_dropout=0.3):
        super(ImprovedActorCriticNetwork, self).__init__()
        self.num_missions = num_missions
        self.num_uavs = num_uavs

        # 총 입력 채널: 미션 좌표(2) + 마스크(1) + UAV 속도(1) + 비용(1) + 예상 도착 시간(1)
        total_in_channels = 2 + 1 + 1 + 1 + 1

        self.gnn_encoder = EnhancedGNNTransformerEncoder(
            in_channels=total_in_channels,
            hidden_channels=gnn_hidden_dim,
            out_channels=embedding_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=gnn_dropout
        )

        # 결합된 특성 크기: 임베딩 + UAV의 2D 좌표 + UAV 속도
        self.combined_feature_size = embedding_dim + 2 + 1

        # 액터 네트워크 동적 레이어 생성
        actor_layers_list = []
        actor_layers_list.append(nn.Linear(self.combined_feature_size, actor_hidden_dim))
        actor_layers_list.append(nn.ReLU())
        actor_layers_list.append(nn.Dropout(actor_dropout))
        for _ in range(actor_layers - 2):
            actor_layers_list.append(nn.Linear(actor_hidden_dim, actor_hidden_dim))
            actor_layers_list.append(nn.ReLU())
            actor_layers_list.append(nn.Dropout(actor_dropout))
        actor_layers_list.append(nn.Linear(actor_hidden_dim, num_missions))
        self.actor_fc = nn.Sequential(*actor_layers_list)

        # 크리틱 네트워크 동적 레이어 생성
        critic_layers_list = []
        critic_layers_list.append(nn.Linear(self.combined_feature_size, critic_hidden_dim))
        critic_layers_list.append(nn.ReLU())
        critic_layers_list.append(nn.Dropout(critic_dropout))
        for _ in range(critic_layers - 2):
            critic_layers_list.append(nn.Linear(critic_hidden_dim, critic_hidden_dim))
            critic_layers_list.append(nn.ReLU())
            critic_layers_list.append(nn.Dropout(critic_dropout))
        critic_layers_list.append(nn.Linear(critic_hidden_dim, 1))
        self.critic_fc = nn.Sequential(*critic_layers_list)

    def forward(self, mission_coords, edge_index, batch, uavs_info, action_mask, speeds, cost_matrix, arrival_times):
        """
        액터-크리틱 네트워크를 통한 순전파.
        
        Args:
            mission_coords (torch.Tensor): 미션의 좌표.
            edge_index (torch.Tensor): GNN을 위한 엣지 인덱스.
            batch (torch.Tensor): 배치 인덱스.
            uavs_info (torch.Tensor): UAV 위치 정보.
            action_mask (torch.Tensor): 액션 마스크.
            speeds (torch.Tensor): UAV 속도.
            cost_matrix (torch.Tensor): 비용 행렬.
            arrival_times (torch.Tensor): 도착 시간 텐서.
        
        Returns:
            tuple: 액션 확률과 상태 값.
        """
        # GNN 인코더를 위한 입력 전처리
        mask_embedded = action_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, self.num_uavs, 1).float()
        speeds_embedded = speeds.unsqueeze(0).repeat(mission_coords.size(0), 1).unsqueeze(-1).float()
        cost_embedded = cost_matrix.T.unsqueeze(-1)
        arrival_times_embedded = arrival_times.T.unsqueeze(-1)
        
        mission_coords_expanded = mission_coords.unsqueeze(1).repeat(1, self.num_uavs, 1)
        mask_weight = 1.0
        speeds_weight = 1.0
        cost_weight = 3.0
        arrival_weight = 1.0
        combined_embedded = torch.cat([mission_coords_expanded,
                                       mask_embedded * mask_weight,
                                       speeds_embedded * speeds_weight,
                                       cost_embedded * cost_weight,
                                       arrival_times_embedded * arrival_weight
                                       ], dim=-1)
        # combined_embedded = torch.cat([mission_coords_expanded, mask_embedded, speeds_embedded, cost_embedded, arrival_times_embedded], dim=-1)
        combined_embedded = combined_embedded.view(-1, combined_embedded.size(-1))
        
        new_batch = batch.repeat_interleave(self.num_uavs)
        
        mission_embeddings = self.gnn_encoder(combined_embedded, edge_index, new_batch)
        
        mission_embeddings_expanded = mission_embeddings.view(mission_coords.size(0), self.num_uavs, -1)
        mission_embeddings_expanded = mission_embeddings_expanded.permute(1, 0, 2).contiguous().view(-1, mission_embeddings_expanded.size(2))
        
        uavs_info_repeated = uavs_info.repeat(mission_coords.size(0), 1)
        speeds_repeated = speeds.repeat(mission_coords.size(0)).unsqueeze(-1)
        
        combined = torch.cat([
            uavs_info_repeated,
            mission_embeddings_expanded,
            speeds_repeated
        ], dim=-1)
        
        # 결합된 특성 크기가 네트워크에 맞도록 조정
        n_features = combined.size(-1)
        combined = combined.view(-1, n_features)
        
        # 액터와 크리틱 네트워크 순전파
        action_logits = self.actor_fc(combined)
        state_values = self.critic_fc(combined)
        
        return action_logits, state_values

# ============================
# 학습 및 검증 함수
# ============================

# 보상 정규화 (에피소드 내에서 정규화)
def normalize_rewards(rewards):
    rewards = torch.tensor(rewards, dtype=torch.float32)
    mean_reward = rewards.mean()
    std_reward = rewards.std() + 1e-5  # 분모가 0이 되는 것을 방지
    return (rewards - mean_reward) / std_reward

def train_model(env, val_env, policy_net, optimizer_actor, optimizer_critic, scheduler_actor, scheduler_critic,
               num_epochs, batch_size, device, edge_index, batch, epsilon_decay, gamma, 
               reward_type='total', alpha=0.5, beta=0.5,
               entropy_coeff=0.01,  # 기본값 설정
               start_epoch=1, checkpoint_path=None, results_path=None, checkpoints_path=None, patience=10, wandb_name="run", epsilon_minimum=0.1):

    # WandB 초기화
    wandb.init(project="multi_uav_mission", name=wandb_name, config={
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate_actor": optimizer_actor.param_groups[0]['lr'],
        "learning_rate_critic": optimizer_critic.param_groups[0]['lr'],
        "weight_decay_actor": optimizer_actor.param_groups[0]['weight_decay'],
        "weight_decay_critic": optimizer_critic.param_groups[0]['weight_decay'],
        "gamma": gamma,
        "patience": patience,
        "gnn_dropout": policy_net.module.gnn_encoder.gnn_output[1].p if isinstance(policy_net, nn.DataParallel) else policy_net.gnn_encoder.gnn_output[1].p,
        "actor_dropout": policy_net.module.actor_fc[-2].p if isinstance(policy_net, nn.DataParallel) else policy_net.actor_fc[-2].p,
        "critic_dropout": policy_net.module.critic_fc[-2].p if isinstance(policy_net, nn.DataParallel) else policy_net.critic_fc[-2].p,
        "num_missions": env.num_missions,
        "num_uavs": env.num_uavs,
        "reward_type": reward_type,
        "alpha": alpha,
        "beta": beta,
        "entropy_coeff": entropy_coeff,  # 엔트로피 가중치 로깅
        "actor_layers": len(policy_net.module.actor_fc) // 3 if isinstance(policy_net, nn.DataParallel) else len(policy_net.actor_fc) // 3,
        "critic_layers": len(policy_net.module.critic_fc) // 3 if isinstance(policy_net, nn.DataParallel) else len(policy_net.critic_fc) // 3,
        "gnn_hidden_dim": policy_net.module.gnn_encoder.layers[0].out_channels // policy_net.module.gnn_encoder.layers[0].heads if isinstance(policy_net, nn.DataParallel) else policy_net.gnn_encoder.layers[0].out_channels // policy_net.gnn_encoder.layers[0].heads,
        "actor_hidden_dim": policy_net.module.actor_fc[0].in_features if isinstance(policy_net, nn.DataParallel) else policy_net.actor_fc[0].in_features,
        "critic_hidden_dim": policy_net.module.critic_fc[0].in_features if isinstance(policy_net, nn.DataParallel) else policy_net.critic_fc[0].in_features
    })
    
    epsilon = 1.5
    epsilon_min = epsilon_minimum

    total_episodes = num_epochs * batch_size
    episode = (start_epoch - 1) * batch_size

    best_validation_reward = -float('inf')
    epochs_no_improve = 0

    try:
        for epoch in tqdm(range(start_epoch, num_epochs + 1), desc="Epochs Progress", position=0):
            # tqdm 인스턴스 생성
            epoch_pbar = tqdm(range(batch_size), desc=f"에폭 {epoch}/{num_epochs}", leave=False)
            for batch_idx in epoch_pbar:
                state = env.reset()
                done = False
                log_probs = []
                values = []
                rewards = []
                entropy_list = []  # 엔트로피를 저장할 리스트
                travel_times = []  # 이동 시간 기록

                while not done:
                    positions = state['positions']
                    uavs_info = positions.to(device)
                    # 임무 완료 여부에 따라 액션 마스크 생성
                    action_mask = create_action_mask(state, done=done)
                    
                    # 비용 행렬과 도착 시간 계산
                    cost_matrix = calculate_cost_matrix(positions, env.missions, env.speeds)
                    arrival_times = calculate_arrival_times(positions, env.missions, env.speeds)

                    # 정책 네트워크 순전파
                    action_probs, state_values = policy_net(
                        env.missions, 
                        edge_index, 
                        batch, 
                        uavs_info, 
                        action_mask, 
                        env.speeds, 
                        cost_matrix,
                        arrival_times
                    )

                    # DataParallel 사용 시 action_probs와 state_values의 첫 번째 차원을 평균으로 병합
                    if isinstance(policy_net, nn.DataParallel):
                        action_probs = action_probs.mean(dim=0)  # 평균을 취하거나 적절한 방식으로 병합
                        state_values = state_values.mean(dim=0)  # 동일하게 병합

                    # UAV 선택 순서 결정
                    uav_order = compute_uav_order(env)
                    
                    # 액션 선택
                    actions = choose_action(action_probs, epsilon, uav_order, global_action_mask=action_mask)

                    # 각 UAV의 액션에 대한 log_prob과 state_value 수집
                    for i, action in enumerate(actions):
                        # 선택된 액션의 확률을 가져옵니다.
                        prob = action_probs[i, action]
                        log_prob = torch.log(prob + 1e-10).squeeze()  # 스칼라로 만듦
                        log_probs.append(log_prob)
                        values.append(state_values[i].squeeze())  # 스칼라로 만듦

                    # 엔트로피 계산 및 저장
                    entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=1).mean()
                    entropy_list.append(entropy)

                    # 환경 스텝
                    next_state, travel_time, done = env.step(actions)

                    # 보상 계산
                    if reward_type == 'max':
                        reward = compute_reward_max_time(env)
                    elif reward_type == 'total':
                        reward = compute_reward_total_time(env)
                    elif reward_type == 'mixed':
                        reward = compute_reward_mixed(env, alpha=alpha, beta=beta)
                    else:
                        raise ValueError(f"Unknown reward_type: {reward_type}")

                    rewards.append(reward)
                    state = next_state

                    # 이동 시간 기록
                    travel_times.append(env.cumulative_travel_times.clone())

                # 여기서 보상을 정규화
                rewards = normalize_rewards(rewards)

                # 할인율을 적용한 누적 보상 계산
                returns = []
                R = 0
                for r in reversed(rewards):
                    R = r + gamma * R
                    returns.insert(0, R)
                returns = torch.tensor(returns, device=device)

                # 보상 표준화 (선택 사항)
                if returns.std() != 0:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-5)

                policy_loss = []
                value_loss = []
                for log_prob, value, R in zip(log_probs, values, returns):
                    advantage = R - value
                    policy_loss.append(-log_prob * advantage)
                    value_loss.append(F.mse_loss(value, R.unsqueeze(0)))

                if policy_loss and value_loss:
                    # 정책 손실과 가치 손실의 평균을 취함
                    policy_loss_total = torch.stack(policy_loss).mean()

                    # 엔트로피 보너스 추가 (탐험 유도)
                    entropy_total = torch.stack(entropy_list).mean()
                    policy_loss_total = policy_loss_total - entropy_coeff * entropy_total  # 엔트로피 가중치 반영

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
                    # policy_loss와 value_loss가 비어 있을 경우, 손실을 0으로 설정
                    loss = torch.tensor(0.0, device=device)

                # epsilon 업데이트
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay

                # 보상과 이동 시간 로깅
                average_travel_time = torch.stack(travel_times).mean().item() if travel_times else 0.0

                # tqdm 진행 표시줄에 정보 업데이트
                epoch_pbar.set_description(f"에폭 {epoch}/{num_epochs} | 배치 {batch_idx+1}/{batch_size} | 보상 {rewards[-1]:.2f} | 손실 {loss.item():.4f} | Epsilon {epsilon:.4f}")

                # WandB에 로그 기록
                wandb.log({
                    "episode": episode,
                    "epoch": epoch,
                    "batch": batch_idx,
                    "policy_loss": policy_loss_total.item() if policy_loss else 0,
                    "value_loss": value_loss_total.item() if value_loss else 0,
                    "loss": loss.item(),
                    "reward": rewards[-1],
                    "epsilon": epsilon,
                    "entropy": entropy_total.item() if policy_loss and value_loss else 0,
                    "average_travel_time": average_travel_time,  # 평균 이동 시간 추가
                    "uav_travel_times": env.cumulative_travel_times.tolist(),  # UAV별 이동 시간 추가
                    "uav_assignments": [len(path) for path in env.paths],  # UAV별 할당된 미션 수
                    "entropy_coeff": entropy_coeff,  # 엔트로피 가중치 로깅
                    "action_probs": wandb.Histogram(action_probs.detach().cpu().numpy())
                })

                episode += 1

            # tqdm 인스턴스 종료
            epoch_pbar.close()

            # 학습률 스케줄러 업데이트
            scheduler_actor.step()
            scheduler_critic.step()

            # 검증
            if epoch % 1 == 0:
                validation_reward = validate_model(val_env, policy_net, device, edge_index, batch, checkpoints_path, results_path, epoch, reward_type, alpha, beta, wandb_name)
                
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
                        'epsilon': epsilon
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
            'epsilon': epsilon
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
        'epsilon': epsilon
    }, last_checkpoint_path)
    print(f"학습이 완료되었습니다. 마지막 체크포인트가 저장되었습니다: {last_checkpoint_path}")

    wandb.finish()


# ============================
# 검증 및 테스트 함수
# ============================

def validate_model(env, policy_net, device, edge_index, batch, checkpoints_path, results_path, epoch, reward_type, alpha, beta, wandb_name="run", temperature=1.0):
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
        wandb_name (str): WandB run 이름.
        temperature (float): 소프트맥스 탐험의 온도 매개변수.
        
    Returns:
        float: 총 검증 보상.
    """
    policy_net.eval()  # 평가 모드로 전환
    state = env.reset()
    done = False
    total_reward = 0
    cumulative_travel_times = torch.zeros(env.num_uavs, device=device)
    paths = [[] for _ in range(env.num_uavs)]

    with torch.no_grad():  # 기울기 계산 비활성화
        while not done:
            positions = state['positions']
            uavs_info = positions.to(device)
            # 임무 완료 여부에 따라 액션 마스크 생성
            action_mask = create_action_mask(state, done=done)
            
            # 비용 행렬과 도착 시간 계산
            cost_matrix = calculate_cost_matrix(positions, env.missions, env.speeds)
            arrival_times = calculate_arrival_times(positions, env.missions, env.speeds)

            # 정책 네트워크 순전파
            action_logits, _ = policy_net(
                env.missions, 
                edge_index, 
                batch, 
                uavs_info, 
                action_mask, 
                env.speeds, 
                cost_matrix,
                arrival_times
            )
            
            # DataParallel 사용 시 action_logits의 첫 번째 차원을 평균으로 병합
            if isinstance(policy_net, nn.DataParallel):
                action_logits = action_logits.mean(dim=0)  # 평균을 취하거나 적절한 방식으로 병합

            # UAV 선택 순서 결정
            uav_order = compute_uav_order(env)
            # 소프트맥스 탐험을 사용하여 액션 선택
            actions = choose_action(action_logits, temperature=temperature, uav_order=uav_order, global_action_mask=action_mask)

            # 환경 스텝
            next_state, travel_time, done = env.step(actions)

            # 보상 계산
            if reward_type == 'max':
                reward = compute_reward_max_time(env)
            elif reward_type == 'total':
                reward = compute_reward_total_time(env)
            elif reward_type == 'mixed':
                reward = compute_reward_mixed(env, alpha=alpha, beta=beta)
            else:
                raise ValueError(f"Unknown reward_type: {reward_type}")

            total_reward += reward
            state = next_state

            for i in range(env.num_uavs):
                paths[i] = env.paths[i]
                cumulative_travel_times[i] = env.cumulative_travel_times[i]

    # 특성 중요도 계산
    feature_importance = compute_feature_importance(
        policy_net, 
        env.missions, 
        edge_index, 
        batch, 
        env.uavs_start,  # UAV 시작 위치를 입력으로 사용
        create_action_mask(env.get_state(), done=done),
        env.speeds,
        calculate_cost_matrix(env.current_positions, env.missions, env.speeds),
        calculate_arrival_times(env.current_positions, env.missions, env.speeds),
        device
    )

    # 검증 모델 저장
    validation_model_save_path = os.path.join(checkpoints_path, f"validation_epoch_{epoch}.pth")
    torch.save(policy_net.state_dict(), validation_model_save_path)
    
    # 가시화
    visualization_path = os.path.join(results_path, f"mission_paths_validation_epoch_{epoch}.png")
    visualize_results(
        env, 
        visualization_path,
        reward=total_reward,
        epsilon=None,
        policy_loss=None,
        value_loss=None,
        folder_name=f"Validation Epoch {epoch} - {wandb_name}"
    )
    
    # WandB에 로그 기록 (보상 및 특성 중요도)
    wandb.log({
        "validation_reward": total_reward,
        "validation_cumulative_travel_times": cumulative_travel_times.tolist(),
        "validation_mission_paths": wandb.Image(visualization_path),
        "epoch": epoch,
        "feature_importance": feature_importance  # 특성 중요도 추가
    })

    policy_net.train()  # 다시 학습 모드로 전환
    
    return total_reward


def test_model(env, policy_net, device, edge_index, batch, checkpoint_path, results_path, reward_type, alpha, beta, wandb_name="run", temperature=1.0):
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
        reward_type (str): 보상 함수 유형.
        alpha (float): 혼합 보상 시 최대 소요 시간 패널티 가중치.
        beta (float): 혼합 보상 시 전체 소요 시간 합 패널티 가중치.
        wandb_name (str): WandB run 이름.
        temperature (float): 소프트맥스 탐험의 온도 매개변수.
    """
    policy_net.eval()  # 평가 모드로 전환

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy_net.load_state_dict(checkpoint['model_state_dict'])
        print(f"체크포인트 '{checkpoint_path}'가 로드되었습니다. 테스트를 시작합니다.")
    else:
        print(f"체크포인트 '{checkpoint_path}'를 찾을 수 없습니다. 테스트를 종료합니다.")
        return

    state = env.reset()
    done = False
    total_reward = 0
    cumulative_travel_times = torch.zeros(env.num_uavs, device=device)
    paths = [[] for _ in range(env.num_uavs)]

    with torch.no_grad():  # 기울기 계산 비활성화
        while not done:
            positions = state['positions']
            uavs_info = positions.to(device)
            # 임무 완료 여부에 따라 액션 마스크 생성
            action_mask = create_action_mask(state, done=done)
            
            # 비용 행렬과 도착 시간 계산
            cost_matrix = calculate_cost_matrix(positions, env.missions, env.speeds)
            arrival_times = calculate_arrival_times(positions, env.missions, env.speeds)

            # 정책 네트워크 순전파
            action_logits, _ = policy_net(
                env.missions, 
                edge_index, 
                batch, 
                uavs_info, 
                action_mask, 
                env.speeds, 
                cost_matrix,
                arrival_times
            )
            
            # DataParallel 사용 시 action_logits의 첫 번째 차원을 평균으로 병합
            if isinstance(policy_net, nn.DataParallel):
                action_logits = action_logits.mean(dim=0)  # 평균을 취하거나 적절한 방식으로 병합

            # UAV 선택 순서 결정
            uav_order = compute_uav_order(env)
            # 소프트맥스 탐험을 사용하여 액션 선택
            actions = choose_action(action_logits, temperature=temperature, uav_order=uav_order, global_action_mask=action_mask)

            # 환경 스텝
            next_state, travel_time, done = env.step(actions)

            # 보상 계산
            if reward_type == 'max':
                reward = compute_reward_max_time(env)
            elif reward_type == 'total':
                reward = compute_reward_total_time(env)
            elif reward_type == 'mixed':
                reward = compute_reward_mixed(env, alpha=alpha, beta=beta)
            else:
                raise ValueError(f"Unknown reward_type: {reward_type}")

            total_reward += reward
            state = next_state

            for i in range(env.num_uavs):
                paths[i] = env.paths[i]
                cumulative_travel_times[i] = env.cumulative_travel_times[i]

    # 특성 중요도 계산
    feature_importance = compute_feature_importance(
        policy_net, 
        env.missions, 
        edge_index, 
        batch, 
        env.uavs_start,  # UAV 시작 위치를 입력으로 사용
        create_action_mask(env.get_state(), done=done),
        env.speeds,
        calculate_cost_matrix(env.current_positions, env.missions, env.speeds),
        calculate_arrival_times(env.current_positions, env.missions, env.speeds),
        device
    )

    print(f"테스트 완료 - 총 보상: {total_reward}")
    visualization_path = os.path.join(results_path, "test_results.png")
    visualize_results(
        env, 
        visualization_path,
        reward=total_reward,
        epsilon=None,
        policy_loss=None,
        value_loss=None,
        folder_name=f"Test - {wandb_name}"
    )
    
    # WandB에 로그 기록 (보상 및 특성 중요도)
    wandb.log({
        "test_reward": total_reward,
        "test_cumulative_travel_times": cumulative_travel_times.tolist(),
        "test_mission_paths": wandb.Image(visualization_path),
        "feature_importance": feature_importance  # 특성 중요도 추가
    })
    
    # 선택 사항: 특성 중요도 시각화 이미지 저장 및 WandB에 업로드
    feature_importance_path = os.path.join(results_path, "feature_importance_test.png")
    visualize_feature_importance(feature_importance, feature_importance_path, title="Feature Importance - Test - " + wandb_name)
    wandb.log({
        "feature_importance_test_bar": wandb.Bar(
            name="Feature Importance - Test",
            x=list(feature_importance.keys()),
            y=list(feature_importance.values())
        ),
        "feature_importance_test_image": wandb.Image(feature_importance_path)
    })

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
        temperature (float, optional): 소프트맥스 탐험의 온도 매개변수.
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
    if epsilon is not None:
        annotations.append(f"Epsilon: {epsilon:.4f}")
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
    if temperature is not None:
        annotations.append(f"Temperature: {temperature:.2f}")
    
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
    import argparse
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from datetime import datetime

    # ============================
    # 인자 파싱 설정
    # ============================
    parser = argparse.ArgumentParser(description="액터-크리틱 GNN을 이용한 다중 UAV 미션 할당 및 최적화")
    parser.add_argument('--config', type=str, help="Path to a json file with configuration parameters")
    parser.add_argument('--gpu', type=str, default='3', help="사용할 GPU 인덱스 (예: '0', '0,1', '0,1,2,3')")
    parser.add_argument('--num_uavs', type=int, default=3, help="UAV의 수")
    parser.add_argument('--num_missions', type=int, default=12, help="미션의 수")
    parser.add_argument('--embedding_dim', type=int, default=64, help="GNN 임베딩 차원")
    parser.add_argument('--gnn_hidden_dim', type=int, default=128, help="GNN 인코더의 숨겨진 차원")
    parser.add_argument('--actor_hidden_dim', type=int, default=128, help="액터 네트워크의 숨겨진 차원")
    parser.add_argument('--critic_hidden_dim', type=int, default=128, help="크리틱 네트워크의 숨겨진 차원")
    parser.add_argument('--actor_layers', type=int, default=6, help="액터 네트워크의 레이어 수")
    parser.add_argument('--critic_layers', type=int, default=6, help="크리틱 네트워크의 레이어 수")
    parser.add_argument('--num_layers', type=int, default=8, help="GNN 레이어 수")
    parser.add_argument('--heads', type=int, default=8, help="GNN Transformer 헤드 수")
    parser.add_argument('--num_epochs', type=int, default=20000, help="에폭 수")
    parser.add_argument('--batch_size', type=int, default=1024, help="배치 크기")
    parser.add_argument('--gamma', type=float, default=0.1, help="할인율 (gamma)")
    parser.add_argument('--lr_actor', type=float, default=3e-4, help="액터 학습률")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="크리틱 학습률")
    parser.add_argument('--weight_decay_actor', type=float, default=1e-4, help="액터 옵티마이저의 weight decay")
    parser.add_argument('--weight_decay_critic', type=float, default=1e-4, help="크리틱 옵티마이저의 weight decay")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="기존 체크포인트의 경로")
    parser.add_argument('--test_mode', action='store_true', help="테스트 모드 활성화")
    parser.add_argument('--train_seed', type=int, default=2024, help="Train 데이터셋 시드는 무조건 랜덤으로 변경됐음.")
    parser.add_argument('--validation_seed', type=int, default=2025, help="Validation 데이터셋 시드")
    parser.add_argument('--test_seed', type=int, default=2026, help="Test 데이터셋 시드")
    parser.add_argument('--time_weight', type=float, default=2.0, help="보상 시간의 가중치")
    parser.add_argument('--lr_step_size', type=int, default=1000, help="학습률 스케줄러의 step size")
    parser.add_argument('--lr_gamma', type=float, default=0.01, help="학습률 스케줄러의 gamma 값")
    parser.add_argument('--entropy_coeff', type=float, default=0.05, help="정책 손실에 추가되는 엔트로피 가중치")
    parser.add_argument('--gnn_dropout', type=float, default=0.3, help="GNN Transformer 인코더의 드롭아웃 비율")
    parser.add_argument('--actor_dropout', type=float, default=0.3, help="액터 네트워크의 드롭아웃 비율")
    parser.add_argument('--critic_dropout', type=float, default=0.3, help="크리틱 네트워크의 드롭아웃 비율")
    
    parser.add_argument('--reward_type', type=str, default='mixed', choices=['max', 'total', 'mixed'], help="보상 함수 유형: 'max', 'total', 'mixed'")
    parser.add_argument('--alpha', type=float, default=0.5, help="혼합 보상 시 최대 소요 시간 패널티 가중치 (reward_type='mixed'일 때 사용)")
    parser.add_argument('--beta', type=float, default=0.5, help="혼합 보상 시 전체 소요 시간 합 패널티 가중치 (reward_type='mixed'일 때 사용)")
    
    parser.add_argument('--use_2opt', action='store_false', help="2-opt 알고리즘을 학습에 포함 여부 확인")
    
    parser.add_argument('--results_dir', type=str, default="/mnt/hdd2/attoman/GNN/results/boltzmann/", help="결과 저장 디렉토리")
    parser.add_argument('--name', type=str, default='boltzmann', help="WandB run name")
    
    parser.add_argument('--temperature', type=float, default=2.0, help="boltzmann 탐험의 온도 매개변수 (tau)")

    # 먼저 --config 인자를 파싱
    args_config, remaining_argv = parser.parse_known_args()

    if args_config.config is not None:
        # JSON 파일에서 파라미터 로드
        with open(args_config.config, 'r') as f:
            config_args = json.load(f)
        # JSON에서 로드한 파라미터를 기본값으로 설정
        parser.set_defaults(**config_args)

    # 나머지 인자 파싱 (커맨드라인 인자가 JSON 파일의 파라미터를 오버라이드)
    args = parser.parse_args()


    # 장치 설정
    # GPU 인자 처리
    if torch.cuda.is_available() and args.gpu:
        gpu_indices = [int(x) for x in args.gpu.split(',')]
        num_gpus = len(gpu_indices)
        if num_gpus > 1:
            device = torch.device(f"cuda:{gpu_indices[0]}")
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_indices))
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

    # 재현성을 위해 시드 설정
    set_seed(args.train_seed)

    # ============================
    # 데이터 생성
    # ============================
    train_data = MissionData(num_missions=args.num_missions, num_uavs=args.num_uavs, seed=args.train_seed, device=device)
    val_data = MissionData(num_missions=args.num_missions, num_uavs=args.num_uavs, seed=args.validation_seed, device=device)
    test_data = MissionData(num_missions=args.num_missions, num_uavs=args.num_uavs, seed=args.test_seed, device=device)

    # ============================
    # 환경 초기화 (2-opt 사용 여부 반영)
    # ============================
    train_env = MissionEnvironment(train_data.missions, train_data.uavs_start, train_data.uavs_speeds, device, mode='train', seed=args.train_seed, time_weight=args.time_weight, use_2opt=args.use_2opt)
    val_env = MissionEnvironment(val_data.missions, val_data.uavs_start, val_data.uavs_speeds, device, mode='val', seed=args.validation_seed, time_weight=args.time_weight, use_2opt=args.use_2opt)
    test_env = MissionEnvironment(test_data.missions, test_data.uavs_start, test_data.uavs_speeds, device, mode='test', seed=args.test_seed, time_weight=args.time_weight, use_2opt=args.use_2opt)

    # ============================
    # edge_index와 batch 생성
    # ============================
    edge_index = create_edge_index(args.num_missions, args.num_uavs).to(device)
    batch = torch.arange(args.num_uavs).repeat_interleave(args.num_missions).to(device)

    # ============================
    # 정책 네트워크 초기화
    # ============================
    policy_net = ImprovedActorCriticNetwork(
        num_missions=args.num_missions,
        num_uavs=args.num_uavs,
        embedding_dim=args.embedding_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        actor_hidden_dim=args.actor_hidden_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        actor_layers=args.actor_layers,
        critic_layers=args.critic_layers,
        num_layers=args.num_layers,
        heads=args.heads,
        gnn_dropout=args.gnn_dropout,
        actor_dropout=args.actor_dropout,
        critic_dropout=args.critic_dropout
    ).to(device)

    # DataParallel 사용 가능 시 적용
    if num_gpus > 1:
        policy_net = nn.DataParallel(policy_net)

    # ============================
    # 옵티마이저 및 스케줄러 초기화
    # ============================
    optimizer_actor = optim.Adam(policy_net.actor_fc.parameters(), lr=args.lr_actor, weight_decay=args.weight_decay_actor)
    optimizer_critic = optim.Adam(policy_net.critic_fc.parameters(), lr=args.lr_critic, weight_decay=args.weight_decay_critic)

    scheduler_actor = optim.lr_scheduler.StepLR(optimizer_actor, step_size=args.lr_step_size, gamma=args.lr_gamma)
    scheduler_critic = optim.lr_scheduler.StepLR(optimizer_critic, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # ============================
    # 결과 및 체크포인트 디렉토리 생성
    # ============================
    num_missions_folder = f"num_missions_{args.num_missions}"


    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"현재 시간: {current_time}")

    # --name 인자를 사용하여 하위 폴더 생성
    name_folder = args.name
    base_dir = os.path.join(args.results_dir, num_missions_folder, current_time, name_folder)
    images_path = os.path.join(base_dir, "images")
    checkpoints_path = os.path.join(base_dir, "checkpoints")
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    # 파라미터를 JSON 파일로 저장
    args_dict = vars(args)
    json_file_path = os.path.join(base_dir, 'config.json')
    with open(json_file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)

    # ============================
    # 학습 또는 테스트 모드 실행
    # ============================
    if args.test_mode:
        test_model(
            env=test_env,
            policy_net=policy_net,
            device=device,
            edge_index=edge_index,
            batch=batch,
            checkpoint_path=args.checkpoint_path,
            results_path=images_path,
            reward_type=args.reward_type,
            alpha=args.alpha,
            beta=args.beta,
            wandb_name=args.name,
            temperature=1.0  # 추가된 매개변수 전달
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
            checkpoint_path=args.checkpoint_path,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            device=device,
            edge_index=edge_index,
            batch=batch,
            gamma=args.gamma,
            reward_type=args.reward_type,
            alpha=args.alpha,
            beta=args.beta,
            entropy_coeff=args.entropy_coeff,
            checkpoint_path=args.checkpoint_path,
            results_path=images_path,
            checkpoints_path=checkpoints_path,
            patience=10,
            wandb_name=args.name,
            temperature=args.temperature  # 추가된 매개변수 전달
        )

if __name__ == "__main__":
    main()
