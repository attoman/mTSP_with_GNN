
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm
import wandb
import argparse

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

# ============================
# 데이터 클래스
# ============================

class MissionData:
    def __init__(self, num_missions=20, num_uavs=3, seed=None, device='cpu'):
        self.num_missions = num_missions
        self.num_uavs = num_uavs
        self.seed = seed
        self.device = device
        self.missions, self.uavs_start, self.uavs_speeds = self.generate_data()

    def generate_data(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
        missions = torch.rand((self.num_missions, 2)) * 100
        start_end_point = missions[0].clone()
        missions[-1] = start_end_point
        uavs_start = start_end_point.unsqueeze(0).repeat(self.num_uavs, 1)
        uavs_speeds = torch.randint(5, 30, (self.num_uavs,), dtype=torch.float)
        return missions.to(self.device), uavs_start.to(self.device), uavs_speeds.to(self.device)


# ============================
# 환경 클래스
# ============================

class MissionEnvironment:
    def __init__(self, missions, uavs_start, uavs_speeds, device='cpu'):
        self.device = device
        self.num_missions = missions.size(0)
        self.num_uavs = uavs_start.size(0)
        self.missions = missions
        self.uavs_start = uavs_start
        self.speeds = uavs_speeds
        self.reset()

    def reset(self):
        self.current_positions = self.uavs_start.clone()
        self.visited = torch.zeros(self.num_missions, dtype=torch.bool, device=self.device)
        self.paths = [[] for _ in range(self.num_uavs)]
        self.cumulative_travel_times = torch.zeros(self.num_uavs, device=self.device)
        self.ready_for_next_action = torch.ones(self.num_uavs, dtype=torch.bool, device=self.device)
        self.targets = [-1] * self.num_uavs
        self.visited[0] = True
        for i in range(self.num_uavs):
            self.paths[i].append(0)
        return self.get_state()

    def get_state(self):
        return {
            'positions': self.current_positions.clone(),
            'visited': self.visited.clone(),
            'ready_for_next_action': self.ready_for_next_action.clone(),
            'targets': self.targets
        }

    def step(self, actions):
        for i, action in enumerate(actions):
            if self.ready_for_next_action[i] and not self.visited[action]:
                self.targets[i] = action
                distance = calculate_distance(self.current_positions[i], self.missions[action])
                travel_time = calculate_travel_time(distance, self.speeds[i].item())
                self.cumulative_travel_times[i] += travel_time
                self.current_positions[i] = self.missions[action]
                self.visited[action] = True
                self.paths[i].append(action)
                self.ready_for_next_action[i] = True
        done = self.visited[1:-1].all()
        return self.get_state(), done


# ============================
# GNN Transformer 인코더
# ============================

class EnhancedGNNTransformerEncoder(nn.Module):
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
    def __init__(self, embedding_dim=64, gnn_hidden_dim=64, actor_hidden_dim=128, critic_hidden_dim=128):
        super(ImprovedActorCriticNetwork, self).__init__()
        self.gnn_encoder = EnhancedGNNTransformerEncoder(
            in_channels=6, hidden_channels=gnn_hidden_dim, out_channels=embedding_dim, num_layers=4, heads=8, dropout=0.3
        )
        self.actor_fc = nn.Sequential(
            nn.Linear(embedding_dim + 3, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, 1)  # Output is 1 action per UAV, adjusted later
        )
        self.critic_fc = nn.Sequential(
            nn.Linear(embedding_dim + 3, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, 1)
        )

    def forward(self, mission_coords, edge_index, batch, uavs_info, action_mask, speeds, dist_matrix, timetogo_matrix):
        mask_embedded = action_mask.unsqueeze(-1).float()
        speeds_embedded = speeds.unsqueeze(-1).unsqueeze(1).repeat(1, mission_coords.size(0), 1)
        dist_embedded = dist_matrix.unsqueeze(-1)
        timetogo_embedded = timetogo_matrix.unsqueeze(-1)
        mission_coords_expanded = mission_coords.unsqueeze(0).repeat(uavs_info.size(0), 1, 1)
        combined_embedded = torch.cat([mission_coords_expanded, mask_embedded, speeds_embedded, dist_embedded, timetogo_embedded], dim=-1)
        combined_embedded = combined_embedded.view(-1, combined_embedded.size(-1))
        new_batch = batch.repeat_interleave(mission_coords.size(0))
        mission_embeddings = self.gnn_encoder(combined_embedded, edge_index, new_batch)
        mission_embeddings = mission_embeddings.view(uavs_info.size(0), -1, mission_embeddings.size(-1))
        uav_embeddings = mission_embeddings.sum(dim=1)
        combined = torch.cat([uavs_info, uav_embeddings, speeds.unsqueeze(-1)], dim=-1)
        action_logits = self.actor_fc(combined)
        action_probs = F.softmax(action_logits, dim=-1)
        state_values = self.critic_fc(combined)
        return action_probs, state_values.squeeze()

# ============================
# 학습 함수
# ============================

def train_model(env, policy_net, optimizer_actor, optimizer_critic, num_epochs, batch_size, device, edge_index, batch):
    wandb.init(project="multi_uav_mission", name="train_run")
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        state = env.reset()
        done = False
        while not done:
            positions = state['positions']
            uavs_info = positions.to(device)
            action_mask = torch.zeros(env.num_uavs, env.num_missions, device=device)
            timetogo_matrix = torch.zeros(env.num_uavs, env.num_missions, device=device)
            dist_matrix = torch.zeros(env.num_uavs, env.num_missions, device=device)
            action_probs, state_values = policy_net(env.missions, edge_index, batch, uavs_info, action_mask, env.speeds, dist_matrix, timetogo_matrix)
            actions = torch.argmax(action_probs, dim=-1).tolist()
            state, done = env.step(actions)
        wandb.log({"epoch": epoch})
    wandb.finish()

# ============================
# 메인 함수
# ============================

def main():
    parser = argparse.ArgumentParser(description="다중 UAV 미션 할당 및 최적화")
    parser.add_argument('--num_uavs', type=int, default=3, help="UAV의 수")
    parser.add_argument('--num_missions', type=int, default=20, help="미션의 수")
    parser.add_argument('--num_epochs', type=int, default=1000, help="에폭 수")
    parser.add_argument('--batch_size', type=int, default=32, help="배치 크기")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = MissionData(num_missions=args.num_missions, num_uavs=args.num_uavs, device=device)
    train_env = MissionEnvironment(train_data.missions, train_data.uavs_start, train_data.uavs_speeds, device)
    edge_index = create_edge_index(args.num_missions, args.num_uavs).to(device)
    batch = torch.arange(args.num_uavs).repeat_interleave(args.num_missions).to(device)

    policy_net = ImprovedActorCriticNetwork().to(device)
    optimizer_actor = optim.Adam(policy_net.actor_fc.parameters(), lr=1e-4)
    optimizer_critic = optim.Adam(policy_net.critic_fc.parameters(), lr=1e-4)
    train_model(train_env, policy_net, optimizer_actor, optimizer_critic, args.num_epochs, args.batch_size, device, edge_index, batch)

if __name__ == "__main__":
    main()
