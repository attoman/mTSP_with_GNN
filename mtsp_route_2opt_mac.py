import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm
import wandb
import logging
import argparse
import sys
import traceback
import matplotlib.pyplot as plt
from datetime import datetime

# ============================
# 유틸리티 함수
# ============================

def calculate_distance(mission1, mission2):
    return torch.sqrt(torch.sum((mission1 - mission2) ** 2))

def calculate_travel_time(distance, speed):
    return distance / speed

def create_edge_index(num_missions):
    adj_matrix = torch.ones((num_missions, num_missions)) - torch.eye(num_missions)
    edge_index = dense_to_sparse(adj_matrix)[0]
    return edge_index

def create_action_mask(state):
    visited = state['visited']
    reserved = state['reserved']
    action_mask = visited | reserved
    return action_mask

# ============================
# 2-opt 알고리즘 함수
# ============================

def two_opt(path, missions):
    best = path
    improved = True
    best_distance = calculate_total_distance(best, missions)
    
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
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

def calculate_total_distance(path, missions):
    distance = 0.0
    for i in range(len(path) - 1):
        mission_from = missions[path[i]]
        mission_to = missions[path[i + 1]]
        distance += calculate_distance(mission_from, mission_to).item()
    return distance

# ============================
# 액션 선택 함수
# ============================

def choose_action(action_probs, epsilon=0.1, uav_order=None):
    actions = []
    action_mask = torch.zeros(action_probs.shape[1], dtype=torch.bool, device=action_probs.device)
    
    if uav_order is None:
        uav_order = list(range(action_probs.shape[0]))
    
    for i in uav_order:
        masked_probs = action_probs[i].clone()
        masked_probs[action_mask] = float('-inf')
        masked_probs = F.softmax(masked_probs, dim=-1)
        
        if random.random() < epsilon:
            valid_actions = torch.where(masked_probs > 0)[0]
            if len(valid_actions) > 0:
                action = random.choice(valid_actions.tolist())
            else:
                action = torch.argmax(masked_probs).item()
        else:
            action = torch.argmax(masked_probs).item()
        
        actions.append(action)
        action_mask[action] = True
    
    return actions

def compute_uav_order(env):
    expected_arrival_times = []
    for i in range(env.num_uavs):
        if env.ready_for_next_action[i]:
            expected_arrival_times.append(0.0)
        else:
            if env.remaining_distances[i] == float('inf'):
                expected_arrival_times.append(float('inf'))
            else:
                expected_time = env.remaining_distances[i].item() / env.speeds[i].item()
                expected_arrival_times.append(expected_time)
    
    uav_indices = list(range(env.num_uavs))
    uav_order = sorted(uav_indices, key=lambda i: expected_arrival_times[i])
    return uav_order

# ============================
# 데이터 클래스
# ============================

class MissionData:
    def __init__(self, num_missions=20, num_uavs=3, seed=42, device='cpu'):
        self.num_missions = num_missions
        self.num_uavs = num_uavs
        self.seed = seed
        self.device = device
        self.missions, self.uavs_start, self.uavs_speeds = self.generate_data()

    def generate_data(self):
        torch.manual_seed(self.seed)
        missions = torch.rand((self.num_missions, 2)) * 100
        missions[-1] = missions[0]
        start_mission = missions[0].unsqueeze(0)
        uavs_start = start_mission.repeat(self.num_uavs, 1)
        uavs_speeds = torch.rand(self.num_uavs) * 9 + 1
        return missions.to(self.device), uavs_start.to(self.device), uavs_speeds.to(self.device)

# ============================
# 강화 학습 환경 클래스 (다중 에이전트)
# ============================

class MissionEnvironment:
    # 생략 (기존 코드와 동일)

# ============================
# GNN Transformer 인코더 및 액터-크리틱 네트워크
# ============================

class GNNTransformerEncoder(nn.Module):
    # 생략 (기존 코드와 동일)

class ActorCriticNetwork(nn.Module):
    # 생략 (기존 코드와 동일)

# ============================
# 학습 루프 및 검증 루프
# ============================

def train_model(env, val_env, policy_net, optimizer, num_epochs, batch_size, device, edge_index, batch, epsilon_decay, start_epoch=1, checkpoint_path=None, results_path=None, checkpoints_path=None):
    # 생략 (기존 코드와 동일)

# ============================
# 검증 및 테스트 함수
# ============================

def validate_model(env, policy_net, device, edge_index, batch, checkpoints_path, results_path, epoch):
    # 생략 (기존 코드와 동일)

def test_model(env, policy_net, device, edge_index, batch, checkpoint_path, results_path):
    # 생략 (기존 코드와 동일)

# ============================
# 시각화 및 결과 저장 함수
# ============================

def visualize_results(env, save_path):
    # 생략 (기존 코드와 동일)

# ============================
# 메인 함수
# ============================

def main():
    parser = argparse.ArgumentParser(description="mTSP Actor-Critic GNN with 2-opt Optimization")
    parser.add_argument('--gpu', type=int, default=0, help="사용할 GPU 인덱스")
    parser.add_argument('--num_uavs', type=int, default=3, help="UAV의 수")
    parser.add_argument('--num_missions', type=int, default=20, help="미션의 수")
    parser.add_argument('--embedding_dim', type=int, default=32, help="GNN 임베딩 차원")
    parser.add_argument('--hidden_dim', type=int, default=128, help="FC 레이어의 은닉 차원")
    parser.add_argument('--num_layers', type=int, default=3, help="GNN 레이어 수")
    parser.add_argument('--heads', type=int, default=4, help="GNN Transformer 헤드 수")
    parser.add_argument('--num_epochs', type=int, default=10000, help="에폭 수")
    parser.add_argument('--batch_size', type=int, default=100, help="배치 크기")
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help="Epsilon 감소율")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="기존 체크포인트의 경로")
    parser.add_argument('--test_mode', action='store_true', help="테스트 모드 활성화")
    parser.add_argument('--validation_seed', type=int, default=43, help="Validation 데이터셋 시드")
    parser.add_argument('--test_seed', type=int, default=44, help="Test 데이터셋 시드")
    parser.add_argument('--results_dir', type=str, default="K:/2024/mTSP/results/2opt/route", help="결과 저장 디렉토리")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS 사용 중")
    else:
        device = torch.device("cpu")
        print("CPU 사용 중")

    # 훈련 데이터 초기화
    train_data = MissionData(num_missions=args.num_missions, num_uavs=args.num_uavs, seed=42, device=device)
    # 검증 데이터 초기화 (다른 시드 사용)
    val_data = MissionData(num_missions=args.num_missions, num_uavs=args.num_uavs, seed=args.validation_seed, device=device)
    # 테스트 데이터 초기화 (또 다른 시드 사용)
    test_data = MissionData(num_missions=args.num_missions, num_uavs=args.num_uavs, seed=args.test_seed, device=device)

    # 환경 생성
    train_env = MissionEnvironment(train_data.missions, train_data.uavs_start, train_data.uavs_speeds, device, mode='train')
    val_env = MissionEnvironment(val_data.missions, val_data.uavs_start, val_data.uavs_speeds, device, mode='val')
    test_env = MissionEnvironment(test_data.missions, test_data.uavs_start, test_data.uavs_speeds, device, mode='test')

    # edge_index 및 batch 생성
    edge_index = create_edge_index(train_data.num_missions).to(device)
    batch = torch.zeros(train_data.num_missions, dtype=torch.long, device=device)

    # 정책 네트워크 및 옵티마이저 초기화
    policy_net = ActorCriticNetwork(
        num_missions=args.num_missions,
        num_uavs=args.num_uavs,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads
    ).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    # 결과 및 체크포인트 경로 설정
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(args.results_dir, "images", current_time)
    checkpoints_path = os.path.join(args.results_dir, "checkpoints", current_time)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    if args.test_mode:
        test_model(env=test_env, 
                   policy_net=policy_net, 
                   device=device, 
                   edge_index=edge_index, 
                   batch=batch, 
                   checkpoint_path=args.checkpoint_path,
                   results_path=results_path)
    else:
        train_model(env=train_env, 
                   val_env=val_env, 
                   policy_net=policy_net, 
                   optimizer=optimizer, 
                   num_epochs=args.num_epochs, 
                   batch_size=args.batch_size, 
                   device=device, 
                   edge_index=edge_index, 
                   batch=batch, 
                   epsilon_decay=args.epsilon_decay, 
                   checkpoint_path=args.checkpoint_path,
                   results_path=results_path,
                   checkpoints_path=checkpoints_path)

if __name__ == "__main__":
    main()
