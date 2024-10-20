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
    """
    2-opt 알고리즘을 사용하여 주어진 경로를 최적화합니다.
    path: 리스트 형태의 미션 인덱스 경로
    missions: 미션 좌표 텐서
    """
    best = path
    improved = True
    best_distance = calculate_total_distance(best, missions)
    
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
                if j - i == 1:  # 인접한 엣지는 스킵
                    continue
                new_path = best[:i] + best[i:j][::-1] + best[j:]
                new_distance = calculate_total_distance(new_path, missions)
                if new_distance < best_distance:
                    best = new_path
                    best_distance = new_distance
                    improved = True
        if improved:
            break  # 첫 번째 개선만 적용하여 빠르게 종료
    return best

def calculate_total_distance(path, missions):
    """
    경로의 총 거리를 계산합니다.
    path: 리스트 형태의 미션 인덱스 경로
    missions: 미션 좌표 텐서
    """
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
    """
    각 UAV가 순차적으로 액션을 선택하도록 수정된 함수.
    한 UAV가 선택한 액션은 이후 UAV의 선택에서 제외됩니다.
    """
    actions = []
    action_mask = torch.zeros(action_probs.shape[1], dtype=torch.bool, device=action_probs.device)  # 전체 미션에 대한 마스크 초기화
    
    if uav_order is None:
        uav_order = list(range(action_probs.shape[0]))
    
    for i in uav_order:
        # 현재 UAV에 대한 액션 확률에 마스크 적용
        masked_probs = action_probs[i].clone()
        masked_probs[action_mask] = float('-inf')  # 이미 선택된 액션을 마스킹
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
        action_mask[action] = True  # 선택된 액션을 마스크에 추가하여 다음 UAV의 선택에서 제외
    
    return actions

def compute_uav_order(env):
    """
    UAV 선택 순서를 결정하는 함수.
    - 준비 상태인 UAV는 우선 순위가 높음.
    - 준비되지 않은 UAV는 남은 거리와 속도를 기반으로 예상 도착 시간을 계산하여 우선 순위 설정.
    """
    expected_arrival_times = []
    for i in range(env.num_uavs):
        if env.ready_for_next_action[i]:
            expected_arrival_times.append(0.0)  # 준비된 UAV는 즉시 선택 가능
        else:
            if env.remaining_distances[i] == float('inf'):
                expected_arrival_times.append(float('inf'))  # 이동 중이지만 현재 목표가 없음
            else:
                expected_time = env.remaining_distances[i].item() / env.speeds[i].item()
                expected_arrival_times.append(expected_time)
    
    uav_indices = list(range(env.num_uavs))
    # 예상 도착 시간이 짧은 순으로 UAV 정렬
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
        # 마지막 미션을 첫 번째 미션과 동일하게 설정하여 시작과 종료 지점이 같도록 함
        missions[-1] = missions[0]
        start_mission = missions[0].unsqueeze(0)
        uavs_start = start_mission.repeat(self.num_uavs, 1)
        uavs_speeds = torch.rand(self.num_uavs) * 9 + 1
        return missions.to(self.device), uavs_start.to(self.device), uavs_speeds.to(self.device)

# ============================
# 강화 학습 환경 클래스 (다중 에이전트)
# ============================

class MissionEnvironment:
    def __init__(self, missions, uavs_start, uavs_speeds, device, mode='train'):
        self.missions = missions
        self.num_missions = missions.size(0)
        self.num_uavs = uavs_start.size(0)
        self.speeds = uavs_speeds
        self.uavs_start = uavs_start
        self.device = device
        self.mode = mode  # 'train', 'val', 'test'
        self.reset()

    def reset(self):
        self.current_positions = self.uavs_start.clone()
        self.visited = torch.zeros(self.num_missions, dtype=torch.bool, device=self.device)
        self.reserved = torch.zeros(self.num_missions, dtype=torch.bool, device=self.device)
        self.paths = [[] for _ in range(self.num_uavs)]
        self.cumulative_travel_times = torch.zeros(self.num_uavs, device=self.device)
        self.ready_for_next_action = torch.ones(self.num_uavs, dtype=torch.bool, device=self.device)
        self.targets = [-1] * self.num_uavs
        self.remaining_distances = torch.full((self.num_uavs,), float('inf'), device=self.device)
        
        # 시작/종료 미션을 이미 방문한 것으로 표시
        self.visited[0] = True
        
        for i in range(self.num_uavs):
            self.paths[i].append(0)
        return self.get_state()

    def get_state(self):
        return {
            'positions': self.current_positions.clone(),
            'visited': self.visited.clone(),
            'reserved': self.reserved.clone(),
            'ready_for_next_action': self.ready_for_next_action.clone(),
            'remaining_distances': self.remaining_distances.clone(),
            'targets': self.targets
        }

    def step(self, actions):
        for i, action in enumerate(actions):
            if self.ready_for_next_action[i] and not self.visited[action] and not self.reserved[action]:
                self.reserved[action] = True
                self.ready_for_next_action[i] = False
                self.targets[i] = action
                mission_from = self.current_positions[i]
                mission_to = self.missions[action]
                self.remaining_distances[i] = calculate_distance(mission_from, mission_to)

        for i, action in enumerate(self.targets):
            if action != -1 and not self.ready_for_next_action[i]:
                distance = self.remaining_distances[i]
                travel_time = calculate_travel_time(distance, self.speeds[i])

                self.cumulative_travel_times[i] += travel_time
                self.current_positions[i] = self.missions[action]
                self.visited[action] = True
                self.paths[i].append(action)
                self.ready_for_next_action[i] = True
                self.reserved[action] = False

        done = self.visited.all()
        if done:
            for i in range(self.num_uavs):
                if not torch.equal(self.current_positions[i], self.missions[-1]):
                    distance = calculate_distance(self.current_positions[i], self.missions[-1])
                    travel_time = calculate_travel_time(distance, self.speeds[i])
                    self.cumulative_travel_times[i] += travel_time
                    self.current_positions[i] = self.missions[-1]
                    self.paths[i].append(self.num_missions - 1)
            total_travel_time = self.cumulative_travel_times.max().item()
            reward = -total_travel_time
        else:
            reward = 0.0

        # 학습 모드일 때만 2-opt 적용
        if self.mode == 'train' and done:
            optimized_paths = []
            optimized_travel_times = torch.zeros(self.num_uavs, device=self.device)
            for i in range(self.num_uavs):
                path = self.paths[i]
                optimized_path = two_opt(path, self.missions)
                optimized_paths.append(optimized_path)
                optimized_travel_times[i] = calculate_total_distance(optimized_path, self.missions)
            # 최적화된 경로의 최대 여행 시간을 보상으로 사용
            optimized_total_travel_time = optimized_travel_times.max().item()
            optimized_reward = -optimized_total_travel_time
            # 보상을 개선된 보상으로 대체
            reward = optimized_reward
            # 원래 경로와 최적화된 경로의 차이를 로그로 남김 (옵션)
            # print(f"최적화 전: {total_travel_time}, 최적화 후: {optimized_total_travel_time}")
        
        return self.get_state(), reward, done

# ============================
# GNN Transformer 인코더 및 액터-크리틱 네트워크
# ============================

class GNNTransformerEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=4):
        super(GNNTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TransformerConv(in_channels, hidden_channels, heads=heads, dropout=0.1))
            in_channels = hidden_channels * heads
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

class ActorCriticNetwork(nn.Module):
    def __init__(self, num_missions, num_uavs, embedding_dim=32, hidden_dim=128, num_layers=3, heads=4):
        super(ActorCriticNetwork, self).__init__()
        self.num_missions = num_missions
        self.num_uavs = num_uavs
        self.gnn_encoder = GNNTransformerEncoder(
            in_channels=2,
            hidden_channels=embedding_dim,
            out_channels=embedding_dim,
            num_layers=num_layers,
            heads=heads
        )
        self.actor_fc = nn.Sequential(
            nn.Linear(embedding_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_missions)
        )
        self.critic_fc = nn.Sequential(
            nn.Linear(embedding_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, mission_coords, edge_index, batch, uavs_info):
        mission_embeddings = self.gnn_encoder(mission_coords, edge_index, batch)
        mission_embeddings_expanded = mission_embeddings.repeat(self.num_uavs, 1)
        combined = torch.cat([uavs_info, mission_embeddings_expanded], dim=-1)

        action_logits = self.actor_fc(combined)
        action_probs = F.softmax(action_logits, dim=-1)
        state_values = self.critic_fc(combined)

        return action_probs, state_values

# ============================
# 학습 루프 및 검증 루프
# ============================

def train_model(env, val_env, policy_net, optimizer, num_epochs, batch_size, device, edge_index, batch, epsilon_decay, start_epoch=1, checkpoint_path=None, results_path=None, checkpoints_path=None):
    wandb.init(project="multi_uav_mission", name="training_run")
    epsilon = 1.0
    epsilon_min = 0.1

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        policy_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        epsilon = checkpoint['epsilon']
        print(f"체크포인트 '{checkpoint_path}'가 로드되었습니다. {start_epoch} 에폭부터 시작합니다.")

    total_episodes = num_epochs * batch_size
    episode = (start_epoch - 1) * batch_size

    try:
        for epoch in tqdm(range(start_epoch, num_epochs + 1), desc="에폭 진행 상황"):
            for batch_idx in range(batch_size):
                state = env.reset()
                done = False
                log_probs = []
                values = []
                rewards = []

                while not done:
                    positions = state['positions']
                    uavs_info = positions.to(device)
                    action_probs, state_values = policy_net(env.missions, edge_index, batch, uavs_info)
                    action_mask = create_action_mask(state)
                    action_probs = action_probs.masked_fill(action_mask, float('-inf'))
                    action_probs = F.softmax(action_probs, dim=-1)
                    
                    # UAV 선택 순서 결정
                    uav_order = compute_uav_order(env)
                    
                    # 수정된 choose_action 사용
                    actions = choose_action(action_probs, epsilon, uav_order)
                    
                    for i in range(env.num_uavs):
                        if action_probs[i].sum() > 0:
                            dist = torch.distributions.Categorical(action_probs[i])
                            log_probs.append(dist.log_prob(torch.tensor(actions[i]).to(device)))

                    next_state, reward, done = env.step(actions)

                    rewards.append(reward)
                    values.append(state_values.squeeze())
                    state = next_state

                # 에피소드 완료 후 2-opt를 이미 적용하여 최적화된 보상을 사용
                R = rewards[-1]
                returns = torch.tensor([R for _ in range(len(values))], device=device)

                if returns.std() != 0:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-5)

                policy_loss = []
                value_loss = []
                for log_prob, value, R in zip(log_probs, values, returns):
                    advantage = R - value
                    policy_loss.append(-log_prob * advantage)
                    value_loss.append(F.mse_loss(value, R.unsqueeze(0)))

                if policy_loss and value_loss:
                    policy_loss_total = torch.stack(policy_loss).sum()
                    value_loss_total = torch.stack(value_loss).sum()
                    loss = policy_loss_total + value_loss_total
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay

                wandb.log({
                    "episode": episode,
                    "epoch": epoch,
                    "batch": batch_idx,
                    "policy_loss": policy_loss_total.item() if policy_loss else 0,
                    "value_loss": value_loss_total.item() if value_loss else 0,
                    "reward": rewards[-1],
                    "epsilon": epsilon
                })

                if episode % 10 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': policy_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epsilon': epsilon
                    }, os.path.join(checkpoints_path, f"episode_{episode}.pth"))
                    # 시각화 결과 저장 (주석 처리됨)
                    visualize_results(env, os.path.join(results_path, f"route_mission_paths_episode_{episode}.png"))
                    wandb.log({"mission_paths": wandb.Image(os.path.join(results_path, f"route_mission_paths_episode_{episode}.png"))})

                episode += 1

            if epoch % 100 == 0:
                validate_model(val_env, policy_net, device, edge_index, batch, checkpoints_path, results_path, epoch)
    except KeyboardInterrupt:
        print("학습이 중단되었습니다. 체크포인트를 저장합니다...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epsilon': epsilon
        }, os.path.join(checkpoints_path, f"interrupted_epoch_{epoch}.pth"))
        print("체크포인트가 저장되었습니다. 안전하게 종료합니다.")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wandb.finish()

# ============================
# 검증 및 테스트 함수
# ============================

def validate_model(env, policy_net, device, edge_index, batch, checkpoints_path, results_path, epoch):
    policy_net.eval()  # 평가 모드로 전환
    state = env.reset()
    done = False
    total_reward = 0
    cumulative_travel_times = torch.zeros(env.num_uavs, device=device)
    paths = [[] for _ in range(env.num_uavs)]

    step_count = 0

    with torch.no_grad():  # 그라디언트 계산 비활성화
        while not done:
            # 상태 출력 (디버깅용)
            print(f"검증 단계 {step_count}: 상태: {state}")
            positions = state['positions']
            uavs_info = positions.to(device)
            action_probs, _ = policy_net(env.missions, edge_index, batch, uavs_info)
            action_mask = create_action_mask(state)
            action_probs = action_probs.masked_fill(action_mask, float('-inf'))
            action_probs = F.softmax(action_probs, dim=-1)
            
            # UAV 선택 순서 결정
            uav_order = compute_uav_order(env)
            
            # 수정된 choose_action 사용, 탐험 비율을 0으로 설정
            actions = choose_action(action_probs, epsilon=0.0, uav_order=uav_order)

            next_state, reward, done = env.step(actions)
            total_reward += reward
            state = next_state

            for i in range(env.num_uavs):
                paths[i] = env.paths[i]
                cumulative_travel_times[i] = env.cumulative_travel_times[i]

            step_count += 1

    # 평가 모드 복귀
    policy_net.train()

    # 검증 모델 저장
    torch.save(policy_net.state_dict(), os.path.join(checkpoints_path, f"validation_epoch_{epoch}.pth"))
    # 시각화 결과 저장
    visualize_results(env, os.path.join(results_path, f"mission_paths_validation_epoch_{epoch}.png"))
    # WandB 로그 기록
    wandb.log({
        "validation_reward": total_reward,
        "validation_cumulative_travel_times": cumulative_travel_times.tolist(),
        "validation_mission_paths": wandb.Image(os.path.join(results_path, f"mission_paths_validation_epoch_{epoch}.png")),
        "epoch": epoch
    })

def test_model(env, policy_net, device, edge_index, batch, checkpoint_path, results_path):
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        policy_net.load_state_dict(checkpoint['model_state_dict'])
        print(f"체크포인트 '{checkpoint_path}'가 로드되었습니다. 테스트를 시작합니다.")
    
    policy_net.eval()  # 평가 모드로 전환
    state = env.reset()
    done = False
    total_reward = 0
    cumulative_travel_times = torch.zeros(env.num_uavs, device=device)
    paths = [[] for _ in range(env.num_uavs)]

    step_count = 0

    with torch.no_grad():  # 그라디언트 계산 비활성화
        while not done:
            # 상태 출력 (디버깅용)
            print(f"테스트 단계 {step_count}: 상태: {state}")
            positions = state['positions']
            uavs_info = positions.to(device)
            action_probs, _ = policy_net(env.missions, edge_index, batch, uavs_info)
            action_mask = create_action_mask(state)
            action_probs = action_probs.masked_fill(action_mask, float('-inf'))
            action_probs = F.softmax(action_probs, dim=-1)
            
            # UAV 선택 순서 결정
            uav_order = compute_uav_order(env)
            
            # 수정된 choose_action 사용, 탐험 비율을 0으로 설정
            actions = choose_action(action_probs, epsilon=0.0, uav_order=uav_order)

            # 선택된 액션 출력 (디버깅용)
            print(f"테스트 단계 {step_count}: 선택된 액션: {actions}")
            next_state, reward, done = env.step(actions)
            total_reward += reward
            state = next_state

            for i in range(env.num_uavs):
                paths[i] = env.paths[i]
                cumulative_travel_times[i] = env.cumulative_travel_times[i]

            step_count += 1

    # 평가 모드 복귀
    policy_net.train()

    print(f"테스트 완료 - 총 보상: {total_reward}")
    # 시각화 결과 저장
    visualize_results(env, os.path.join(results_path, "test_results.png"))
    plt.show()

# ============================
# 시각화 및 결과 저장 함수
# ============================

def visualize_results(env, save_path):
    plt.figure(figsize=(10, 10))
    missions = env.missions.cpu().numpy()
    plt.scatter(missions[:, 0], missions[:, 1], c='blue', marker='o', label='Mission')
    plt.scatter(missions[0, 0], missions[0, 1], c='green', marker='s', s=100, label='Start/End Point')
    
    for i, path in enumerate(env.paths):
        path_coords = missions[path]
        plt.plot(path_coords[:, 0], path_coords[:, 1], marker='x', label=f'UAV {i} Path (Velocity: {env.speeds[i].item():.2f})')
    
    plt.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('UAV MTSP')
    plt.savefig(save_path)
    plt.close()

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
    parser.add_argument('--test_seed', type=int, default=44, help="Test 데이터셋 시드")  # 테스트 세트용 시드 추가
    parser.add_argument('--results_dir', type=str, default="D:/mTSP/results/2opt/route", help="결과 저장 디렉토리")  # 유연성을 위한 결과 디렉토리 추가
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        print(f"GPU {args.gpu} 사용 중: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("CPU 사용 중")

    # 훈련 데이터 초기화
    train_data = MissionData(num_missions=args.num_missions, num_uavs=args.num_uavs, seed=42, device=device)
    # 검증 데이터 초기화 (다른 시드 사용)
    val_data = MissionData(num_missions=args.num_missions, num_uavs=args.num_uavs, seed=args.validation_seed, device=device)
    # 테스트 데이터 초기화 (또 다른 시드 사용)
    test_data = MissionData(num_missions=args.num_missions, num_uavs=args.num_uavs, seed=args.test_seed, device=device)  # 테스트 세트용 데이터 추가

    # 환경 생성
    train_env = MissionEnvironment(train_data.missions, train_data.uavs_start, train_data.uavs_speeds, device, mode='train')
    val_env = MissionEnvironment(val_data.missions, val_data.uavs_start, val_data.uavs_speeds, device, mode='val')
    test_env = MissionEnvironment(test_data.missions, test_data.uavs_start, test_data.uavs_speeds, device, mode='test')  # 테스트 환경 추가

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
        # 테스트 모드일 경우, 테스트 환경을 사용하여 모델 평가
        test_model(env=test_env, 
                   policy_net=policy_net, 
                   device=device, 
                   edge_index=edge_index, 
                   batch=batch, 
                   checkpoint_path=args.checkpoint_path,
                   results_path=results_path)
    else:
        # 훈련 모드일 경우, 훈련 환경에서 모델 학습 및 검증 환경에서 검증
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
        # 훈련 후 테스트를 원할 경우, 아래 주석을 해제하여 테스트 수행
        # test_model(env=test_env, 
        #            policy_net=policy_net, 
        #            device=device, 
        #            edge_index=edge_index, 
        #            batch=batch, 
        #            checkpoint_path=None,  # 필요시 체크포인트 지정
        #            results_path=results_path)

if __name__ == "__main__":
    main()
