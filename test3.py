import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import math
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm
import wandb
import argparse
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union, Any

# 의존성 확인
required_packages = {
    "torch_geometric": "pip install torch-geometric",
    "wandb": "pip install wandb"
}
for package, install_cmd in required_packages.items():
    try:
        __import__(package)
    except ImportError:
        raise ImportError(f"{package}가 설치되어 있어야 합니다. '{install_cmd}'를 실행하세요.")

# ### 설정 클래스
@dataclass
class TrainingConfig:
    """학습 매개변수 관리"""
    num_epochs: int = 100
    batch_size: int = 32
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    temperature: float = 1.0
    temperature_min: float = 0.1
    temperature_decay: float = 0.995
    reward_type: str = 'mixed'
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.1
    delta: float = 0.2
    checkpoint_interval: int = 10
    validation_interval: int = 5
    gradient_clip: float = 1.0
    early_stopping_patience: int = 15
    warmup_steps: int = 1000
    use_curriculum: bool = True
    curriculum_min_missions: int = 5
    hidden_dim: int = 128
    num_gat_layers: int = 3
    gat_heads: int = 8
    dropout: float = 0.1
    transformer_nhead: int = 8
    transformer_num_layers: int = 3
    max_sequence_length: int = 100
    max_step_limit: int = 100
    risk_penalty: float = 10.0

# ### 유틸리티 함수
def calculate_distance(mission1: torch.Tensor, mission2: torch.Tensor) -> torch.Tensor:
    """두 미션 간 유클리드 거리 계산"""
    return torch.sqrt(torch.sum((mission1 - mission2) ** 2) + 1e-8)

def calculate_distance_matrix(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
    """두 점 집합 간 거리 행렬 계산"""
    p1 = points1.unsqueeze(1)
    p2 = points2.unsqueeze(0)
    return torch.sqrt(torch.sum((p1 - p2) ** 2, dim=2) + 1e-8)

def dynamic_chunk_size(n_positions, n_targets, n_circles, device):
    """동적 청크 크기 계산"""
    if device.type == 'cuda':
        mem_available = torch.cuda.memory_reserved() / (1024 ** 2)  # MB 단위
        target_mem = min(500, mem_available * 0.8)  # 80% 사용 목표
        chunk_size = int(target_mem / (n_targets * n_circles * 8 / 1024))  # 8바이트 기준
    else:
        chunk_size = 100
    return max(10, min(chunk_size, n_positions))

def compute_segment_circle_intersections(positions: torch.Tensor, targets: torch.Tensor,
                                        circle_centers: torch.Tensor, circle_radii: torch.Tensor,
                                        device: torch.device) -> torch.Tensor:
    """선분-원 교차 계산 (동적 청크 단위로 메모리 효율성 개선)"""
    n_positions = positions.shape[0]
    n_targets = targets.shape[0]
    n_circles = circle_centers.shape[0] if len(circle_centers.shape) > 0 else 0
    if n_circles == 0:
        return torch.zeros((n_positions, n_targets), dtype=torch.bool, device=device)
    chunk_size = dynamic_chunk_size(n_positions, n_targets, n_circles, device)
    intersections = torch.zeros((n_positions, n_targets), dtype=torch.bool, device=device)
    for start in range(0, n_positions, chunk_size):
        end = min(start + chunk_size, n_positions)
        A_chunk = positions[start:end].unsqueeze(1)
        d_chunk = targets.unsqueeze(0) - A_chunk
        for c_idx in range(n_circles):
            center = circle_centers[c_idx]
            radius = circle_radii[c_idx]
            C = center.view(1, 1, -1)
            f_chunk = A_chunk - C
            a = (d_chunk ** 2).sum(dim=2)
            b = 2 * (f_chunk * d_chunk).sum(dim=2)
            c = (f_chunk ** 2).sum(dim=2) - radius**2
            discriminant = b**2 - 4*a*c
            valid_discriminant = discriminant >= 0
            valid_a = a != 0
            mask = valid_discriminant & valid_a
            if mask.any():
                sqrt_disc = discriminant[mask].sqrt()
                t1 = (-b[mask] - sqrt_disc) / (2 * a[mask])
                t2 = (-b[mask] + sqrt_disc) / (2 * a[mask])
                t1_valid = (0 <= t1) & (t1 <= 1)
                t2_valid = (0 <= t2) & (t2 <= 1)
                intersections[start:end] |= (t1_valid | t2_valid).any(dim=1).view(-1, n_targets)
    return intersections

def create_edge_index(num_missions: int) -> torch.Tensor:
    """그래프 엣지 인덱스 생성"""
    adj = torch.ones((num_missions, num_missions))
    edge_index, _ = dense_to_sparse(adj)
    return edge_index

def precompute_edge_indices(max_missions: int, device: torch.device) -> Dict[int, torch.Tensor]:
    """엣지 인덱스 캐싱"""
    cache = {}
    for n in range(2, max_missions + 1):
        cache[n] = create_edge_index(n).to(device)
    return cache

def get_subsequent_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Transformer 디코더용 후속 마스크 생성"""
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

# ### 데이터 및 환경 클래스
class MissionData:
    """미션 데이터 생성"""
    def __init__(self, num_missions: int = 20, num_uavs: int = 3, seed: Optional[int] = None,
                 device: torch.device = torch.device('cpu')):
        self.num_missions = num_missions
        self.num_uavs = num_uavs
        self.seed = seed
        self.device = device
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
        self.missions, self.uavs_start, self.uavs_end, self.uavs_speeds, self.uav_types = self._generate_mission_data()
        self.risk_areas = self._generate_risk_areas()
        self.no_entry_zones = self._generate_no_entry_zones()
        self.risk_centers, self.risk_radii = self._create_area_tensors(self.risk_areas)
        self.zone_centers, self.zone_radii = self._create_area_tensors(self.no_entry_zones)

    def _generate_mission_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """미션 및 UAV 데이터 생성"""
        missions = torch.rand((self.num_missions - 2, 2), device=self.device) * 100
        start_point = torch.rand((1, 2), device=self.device) * 100
        end_point = torch.rand((1, 2), device=self.device) * 100
        missions = torch.cat([start_point, missions, end_point], dim=0)
        uavs_start = start_point.repeat(self.num_uavs, 1)
        uavs_end = end_point.repeat(self.num_uavs, 1)

        # UAV 유형 및 속도 설정 (고정익: 0, 회전익: 1)
        num_fixed = max(1, self.num_uavs // 2)
        num_rotary = self.num_uavs - num_fixed
        uav_types = torch.cat([torch.zeros(num_fixed), torch.ones(num_rotary)]).to(self.device)
        fixed_speeds = torch.randint(30, 60, (num_fixed,), device=self.device).float()
        rotary_speeds = torch.randint(5, 15, (num_rotary,), device=self.device).float()
        uavs_speeds = torch.cat([fixed_speeds, rotary_speeds])
        return missions, uavs_start, uavs_end, uavs_speeds, uav_types

    def _generate_risk_areas(self) -> List[Dict[str, Union[torch.Tensor, float]]]:
        """위험 지역 생성"""
        num_risk_areas = random.randint(1, 5)
        return [{'center': torch.rand(2, device=self.device) * 100, 'radius': random.uniform(5, 15)}
                for _ in range(num_risk_areas)]

    def _generate_no_entry_zones(self) -> List[Dict[str, Union[torch.Tensor, float]]]:
        """출입 불가 지역 생성"""
        num_no_entry_zones = random.randint(1, 3)
        return [{'center': torch.rand(2, device=self.device) * 100, 'radius': random.uniform(3, 10)}
                for _ in range(num_no_entry_zones)]

    def _create_area_tensors(self, areas: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """영역 텐서 생성"""
        if not areas:
            return torch.empty((0, 2), device=self.device), torch.empty(0, device=self.device)
        centers = torch.stack([area['center'] for area in areas])
        radii = torch.tensor([area['radius'] for area in areas], device=self.device)
        return centers, radii

class MissionEnvironment:
    """다중 UAV 미션 환경"""
    def __init__(self, missions: torch.Tensor, uavs_start: torch.Tensor, uavs_end: torch.Tensor,
                 uavs_speeds: torch.Tensor, uav_types: torch.Tensor, risk_centers: torch.Tensor,
                 risk_radii: torch.Tensor, zone_centers: torch.Tensor, zone_radii: torch.Tensor,
                 device: torch.device, seed: Optional[int] = None, curriculum_epoch: Optional[int] = None,
                 total_epochs: Optional[int] = None, min_missions: int = 5):
        self.device = device
        self.seed = seed
        self.original_missions = missions.clone()
        self.max_missions = missions.size(0)
        self.uavs_start = uavs_start
        self.uavs_end = uavs_end
        self.speeds = uavs_speeds
        self.uav_types = uav_types
        self.risk_centers = risk_centers
        self.risk_radii = risk_radii
        self.zone_centers = zone_centers
        self.zone_radii = zone_radii
        self.use_curriculum = curriculum_epoch is not None and total_epochs is not None
        self.curriculum_epoch = curriculum_epoch
        self.total_epochs = total_epochs
        self.min_missions = min_missions
        self.reset()

    def adjust_curriculum(self) -> int:
        """커리큘럼에 따른 미션 수 조정"""
        if not self.use_curriculum:
            return self.max_missions
        progress = min(1.0, self.curriculum_epoch / (self.total_epochs * 0.8))
        num_missions = int(self.min_missions + (self.max_missions - self.min_missions) * progress)
        return min(max(self.min_missions, num_missions), self.max_missions)

    def reset(self) -> Dict[str, torch.Tensor]:
        """환경 초기화"""
        if self.use_curriculum:
            num_missions = self.adjust_curriculum()
            selected_indices = [0]
            middle_count = num_missions - 2
            if middle_count > 0:
                middle_indices = random.sample(range(1, self.max_missions - 1), min(middle_count, self.max_missions - 2))
                selected_indices.extend(sorted(middle_indices))
            selected_indices.append(self.max_missions - 1)
            self.missions = self.original_missions[selected_indices].clone()
        else:
            self.missions = self.original_missions.clone()

        self.num_missions = self.missions.size(0)
        self.num_uavs = self.uavs_start.size(0)
        self.current_positions = self.uavs_start.clone()
        self.visited = torch.zeros(self.num_missions, dtype=torch.bool, device=self.device)
        self.visited[0] = True
        self.reserved = torch.zeros_like(self.visited)
        self.paths = [[] for _ in range(self.num_uavs)]
        self.cumulative_travel_times = torch.zeros(self.num_uavs, device=self.device)
        self.ready_for_next_action = torch.ones(self.num_uavs, dtype=torch.bool, device=self.device)
        self.targets = torch.full((self.num_uavs,), -1, dtype=torch.long, device=self.device)
        self.remaining_distances = torch.full((self.num_uavs,), float('inf'), device=self.device)
        self.assigned_missions = [[] for _ in range(self.num_uavs)]
        return self.get_state()

    def get_state(self) -> Dict[str, torch.Tensor]:
        """현재 상태 반환"""
        return {
            'positions': self.current_positions.clone(),
            'visited': self.visited.clone(),
            'reserved': self.reserved.clone(),
            'ready_for_next_action': self.ready_for_next_action.clone(),
            'cumulative_times': self.cumulative_travel_times.clone(),
            'targets': self.targets.clone(),
            'missions': self.missions.clone()
        }

    def create_action_mask(self) -> torch.Tensor:
        """액션 마스크 생성"""
        mask_base = (self.visited | self.reserved).unsqueeze(0).repeat(self.num_uavs, 1)
        if not self.visited[1:-1].any():
            mask_base[:, -1] = True
        if self.visited[1:-1].all():
            mask_base[:, -1] = False
        mask_base[~self.ready_for_next_action] = True

        if self.zone_centers.shape[0] > 0:
            ready_uavs = torch.where(self.ready_for_next_action)[0]
            if len(ready_uavs) > 0:
                ready_positions = self.current_positions[ready_uavs]
                intersections = compute_segment_circle_intersections(
                    ready_positions, self.missions, self.zone_centers, self.zone_radii, self.device
                )
                for i, u_idx in enumerate(ready_uavs):
                    mask_base[u_idx] |= intersections[i]

        # 항상 유효한 액션 제공 (시작점 복귀)
        for u in range(self.num_uavs):
            if mask_base[u].all():
                mask_base[u, 0] = False  # 시작점 항상 유효
        return mask_base

    def calculate_cost_matrix(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """시간 기반 비용 행렬 계산"""
        dist_matrix = calculate_distance_matrix(self.current_positions, self.missions)
        speeds_expanded = self.speeds.unsqueeze(1)
        timetogo_matrix = dist_matrix / (speeds_expanded + 1e-5)

        if self.risk_centers.shape[0] > 0:
            intersections = compute_segment_circle_intersections(
                self.current_positions, self.missions, self.risk_centers, self.risk_radii, self.device
            )
            timetogo_matrix += intersections.float() * 10.0
        return timetogo_matrix, dist_matrix

    def step(self, actions: List[int]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, bool]:
        """액션 실행 및 다음 상태 반환"""
        actions = torch.tensor(actions, device=self.device)
        travel_times = torch.zeros(self.num_uavs, device=self.device)

        for u, action in enumerate(actions):
            if not self.ready_for_next_action[u] or action == -1:
                continue
            if not self.visited[action] and not self.reserved[action]:
                self.reserved[action] = True
                self.ready_for_next_action[u] = False
                self.targets[u] = action
                self.remaining_distances[u] = calculate_distance(self.current_positions[u], self.missions[action])

        for u, target in enumerate(self.targets):
            if target != -1 and not self.ready_for_next_action[u]:
                travel_time = self.remaining_distances[u] / (self.speeds[u] + 1e-5)
                self.cumulative_travel_times[u] += travel_time
                self.current_positions[u] = self.missions[target]
                if target != self.num_missions - 1:
                    self.visited[target] = True
                self.assigned_missions[u].append(target.item())
                self.ready_for_next_action[u] = True
                self.reserved[target] = False
                self.targets[u] = -1
                travel_times[u] = travel_time

        all_mid_missions_done = self.visited[1:-1].all().item()
        if all_mid_missions_done:
            for u in range(self.num_uavs):
                if self.ready_for_next_action[u] and self.targets[u] != self.num_missions - 1 and not self.visited[-1]:
                    self.targets[u] = self.num_missions - 1
                    self.ready_for_next_action[u] = False
                    self.remaining_distances[u] = calculate_distance(self.current_positions[u], self.missions[-1])

        done = all_mid_missions_done and any(self.num_missions - 1 in path for path in self.assigned_missions)
        return self.get_state(), travel_times, done

# ### 신경망 정의
class PositionalEncoding(nn.Module):
    """Transformer 위치 인코딩"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class GraphEncoder(nn.Module):
    """그래프 인코더"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer, norm in zip(self.gat_layers, self.layer_norms):
            x_res = x
            x = layer(x, edge_index)
            x = F.relu(x)
            x = norm(x + x_res)
        return x

class MissionEmbedding(nn.Module):
    """미션 및 UAV 임베딩"""
    def __init__(self, input_dim: int, hidden_dim: int, num_missions: int, dropout: float = 0.1):
        super().__init__()
        self.mission_embedding = nn.Linear(input_dim, hidden_dim)
        self.mission_id_embedding = nn.Embedding(num_missions, hidden_dim)
        self.uav_embedding = nn.Linear(input_dim + 1, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, missions: torch.Tensor, uav_positions: torch.Tensor, uav_speeds: torch.Tensor,
                mission_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if mission_ids is None:
            mission_ids = torch.arange(missions.size(0), device=missions.device)
        mission_coord_emb = self.mission_embedding(missions)
        mission_id_emb = self.mission_id_embedding(mission_ids)
        mission_emb = mission_coord_emb + mission_id_emb
        uav_features = torch.cat([uav_positions, uav_speeds.unsqueeze(1)], dim=1)
        uav_emb = self.uav_embedding(uav_features)
        return self.dropout(mission_emb), self.dropout(uav_emb)

class TransformerDecoder(nn.Module):
    """Transformer 디코더"""
    def __init__(self, d_model: int, nhead: int, num_layers: int = 3, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return self.norm(output)

class TransformerAllocationNetwork(nn.Module):
    """Transformer 기반 액터 네트워크"""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_gat_layers: int = 3, gat_heads: int = 8,
                 num_transformer_layers: int = 3, transformer_heads: int = 8, max_missions: int = 100, dropout: float = 0.1):
        super().__init__()
        self.embedding = MissionEmbedding(input_dim, hidden_dim, max_missions, dropout)
        self.graph_encoder = GraphEncoder(hidden_dim, hidden_dim, num_gat_layers, gat_heads, dropout)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        self.transformer_decoder = TransformerDecoder(hidden_dim, transformer_heads, num_transformer_layers,
                                                    hidden_dim * 4, dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, missions: torch.Tensor, edge_index: torch.Tensor, uav_positions: torch.Tensor,
                uav_speeds: torch.Tensor, assigned_missions: List[List[int]], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        num_missions = missions.size(0)
        num_uavs = uav_positions.size(0)
        device = missions.device

        mission_emb, uav_emb = self.embedding(missions, uav_positions, uav_speeds)
        node_types = torch.zeros(num_missions, dtype=torch.long, device=device)
        mission_encoding = self.graph_encoder(mission_emb, edge_index, node_types)

        max_seq_len = max([len(seq) for seq in assigned_missions], default=0) + 1
        seq_tensor = torch.full((num_uavs, max_seq_len), -1, dtype=torch.long, device=device)
        for i, seq in enumerate(assigned_missions):
            if seq:
                seq_tensor[i, :len(seq)] = torch.tensor(seq, device=device)

        padding_mask = (seq_tensor == -1)
        tgt = torch.zeros(num_uavs, max_seq_len, hidden_dim, device=device)
        for i in range(num_uavs):
            tgt[i, 0] = uav_emb[i]
            for j in range(1, max_seq_len):
                if not padding_mask[i, j]:
                    mission_idx = seq_tensor[i, j-1].item()
                    if 0 <= mission_idx < num_missions:
                        tgt[i, j] = mission_encoding[mission_idx]

        tgt = self.pos_encoder(tgt)
        tgt_mask = get_subsequent_mask(max_seq_len, device)
        memory = mission_encoding.unsqueeze(0).expand(num_uavs, -1, -1)
        decoder_output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask)

        seq_lens = max_seq_len - padding_mask.sum(dim=1)
        queries = torch.zeros(num_uavs, hidden_dim, device=device)
        for i in range(num_uavs):
            idx = max(0, seq_lens[i]-1)
            queries[i] = decoder_output[i, idx]

        logits = torch.matmul(queries, mission_encoding.T)
        if mask is not None:
            logits = logits.masked_fill(mask, -1e9)
        return logits

class SurrogateNetwork(nn.Module):
    """크리틱 네트워크"""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_layers: int = 3, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.embedding_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.cost_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_types: torch.Tensor, batch=None) -> torch.Tensor:
        x = self.input_proj(x)
        for layer, norm in zip(self.embedding_layers, self.layer_norms):
            x_res = x
            x = layer(x, edge_index)
            x = F.relu(x)
            x = norm(x + x_res)
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, batch)
        return self.cost_predictor(x)

class TransformerActorCriticNetwork(nn.Module):
    """액터-크리틱 네트워크"""
    def __init__(self, max_missions: int, num_uavs: int, hidden_dim: int = 128, num_gat_layers: int = 3,
                 gat_heads: int = 8, dropout: float = 0.1, transformer_heads: int = 8, transformer_layers: int = 3):
        super().__init__()
        self.actor = TransformerAllocationNetwork(2, hidden_dim, num_gat_layers, gat_heads, transformer_layers,
                                                 transformer_heads, max_missions, dropout)
        self.critic = SurrogateNetwork(2, hidden_dim, num_gat_layers, gat_heads, dropout)

    def forward(self, missions: torch.Tensor, edge_index: torch.Tensor, batch, uav_positions: torch.Tensor,
                uav_speeds: torch.Tensor, action_mask: torch.Tensor, assigned_missions: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        action_logits = self.actor(missions, edge_index, uav_positions, uav_speeds, assigned_missions, action_mask)
        node_types = torch.zeros(missions.size(0), dtype=torch.long, device=missions.device)
        state_value = self.critic(missions, edge_index, node_types, batch)
        return action_logits, state_value

# ### 학습 보조 함수
def choose_action(action_logits: torch.Tensor, temperature: float, action_mask: torch.Tensor) -> List[int]:
    """액션 샘플링"""
    probs = F.softmax(action_logits / temperature, dim=-1)
    actions = []
    for i in range(action_logits.size(0)):
        if action_mask[i].all():
            actions.append(0)  # 시작점 복귀
        else:
            action = torch.multinomial(probs[i], 1).item()
            actions.append(action)
    return actions

def compute_episode_reward(env: MissionEnvironment, config: TrainingConfig) -> torch.Tensor:
    """MISOCP 형태의 시간 기반 보상 계산"""
    travel_times = env.cumulative_travel_times
    total_time = travel_times.sum()
    time_std = travel_times.std() if travel_times.size(0) > 1 else torch.tensor(1e-8, device=env.device)
    max_time = travel_times.max()
    balance_penalty = torch.sum((torch.tensor([len(m) for m in env.assigned_missions], device=env.device) - env.num_missions / env.num_uavs) ** 2)
    return -(config.alpha * total_time / env.num_uavs + config.beta * (time_std / (total_time / env.num_uavs)) + config.gamma * (max_time / (total_time / env.num_uavs) - 1) + 0.5 * balance_penalty)

class WarmupScheduler:
    """워밍업 학습률 스케줄러"""
    def __init__(self, optimizer, warmup_steps, scheduler):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.scheduler = scheduler
        self.current_step = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr_scale = min(1.0, self.current_step / self.warmup_steps)
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * lr_scale
        else:
            self.scheduler.step()

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

def train_model(env: MissionEnvironment, val_env: MissionEnvironment, policy_net: TransformerActorCriticNetwork,
               optimizer_actor: optim.Optimizer, optimizer_critic: optim.Optimizer, scheduler_actor, scheduler_critic,
               device: torch.device, edge_indices_cache: Dict[int, torch.Tensor], config: TrainingConfig,
               checkpoint_path: Optional[str] = None, results_dir: str = "./results") -> None:
    """모델 학습"""
    run_name = f"multi_uav_mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project="multi_uav_mission", name=run_name, config=vars(config))

    os.makedirs(results_dir, exist_ok=True)
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    temperature = config.temperature
    best_reward = -float('inf')
    no_improvement_count = 0
    global_step = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            policy_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
            optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            temperature = checkpoint.get('temperature', temperature)
            best_reward = checkpoint.get('best_reward', -float('inf'))
            global_step = checkpoint.get('global_step', 0)
            print(f"체크포인트 로드: {checkpoint_path}")
        except Exception as e:
            logging.error(f"Checkpoint load failed: {str(e)}")
            print(f"체크포인트 로드 실패: {e}. 기본 모델로 초기화합니다.")
            policy_net = TransformerActorCriticNetwork(config.max_missions, config.num_uavs).to(device)
            optimizer_actor = optim.Adam(policy_net.actor.parameters(), lr=config.lr_actor)
            optimizer_critic = optim.Adam(policy_net.critic.parameters(), lr=config.lr_critic)

    else:
        start_epoch = 1

    for epoch in tqdm(range(start_epoch, config.num_epochs + 1), desc="에포크"):
        if config.use_curriculum:
            env.curriculum_epoch = epoch
            val_env.curriculum_epoch = epoch

        epoch_losses = []
        epoch_rewards = []

        for _ in tqdm(range(config.batch_size), desc=f"에포크 {epoch}", leave=False):
            state = env.reset()
            done = False
            log_probs = []
            values = []
            rewards = []

            num_missions = env.num_missions
            if num_missions not in edge_indices_cache:
                edge_indices_cache[num_missions] = create_edge_index(num_missions).to(device)
            edge_index = edge_indices_cache[num_missions]
            batch = torch.zeros(num_missions, dtype=torch.long, device=device)

            step_count = 0
            while not done and step_count < config.max_step_limit:
                step_count += 1
                global_step += 1

                action_mask = env.create_action_mask()
                action_logits, state_values = policy_net(
                    env.missions, edge_index, batch, state['positions'], env.speeds,
                    action_mask, env.assigned_missions
                )

                actions = choose_action(action_logits, temperature, action_mask)
                for i, action in enumerate(actions):
                    if action != -1:
                        probs = F.softmax(action_logits[i], dim=-1)
                        log_probs.append(torch.log(probs[action] + 1e-10))
                        values.append(state_values)

                next_state, _, done = env.step(actions)
                reward = compute_episode_reward(env, config)
                rewards.append(reward)
                state = next_state

            if not log_probs:
                continue

            R = torch.stack(rewards).sum()
            epoch_rewards.append(R.item())
            returns = torch.full((len(values),), R, device=device)
            values = torch.stack(values).squeeze()
            advantage = returns - values.detach()
            if advantage.numel() > 1:
                if torch.isnan(advantage).any():
                    print(f"Warning: NaN detected in advantage at step {global_step}")
                    continue
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            policy_loss = torch.stack([-lp * adv for lp, adv in zip(log_probs, advantage)]).mean()
            value_loss = F.mse_loss(values, returns)
            entropy_coef = max(0.001, 0.01 * (1 - epoch / config.num_epochs))
            entropy = -(F.softmax(action_logits, dim=-1) * F.log_softmax(action_logits, dim=-1)).sum(dim=-1).mean()
            loss = policy_loss + value_loss - entropy_coef * entropy
            epoch_losses.append(loss.item())

            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), config.gradient_clip)
            optimizer_actor.step()
            optimizer_critic.step()

            if global_step % 100 == 0:
                wandb.log({
                    "global_step": global_step,
                    "loss": loss.item(),
                    "reward": R.item(),
                    "temperature": temperature
                })

        temperature = max(temperature * config.temperature_decay, config.temperature_min)
        scheduler_actor.step()
        scheduler_critic.step()

        if epoch % config.validation_interval == 0:
            val_reward = sum([compute_episode_reward(val_env, config) for _ in range(5)]) / 5
            if val_reward > best_reward:
                best_reward = val_reward
                no_improvement_count = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': policy_net.state_dict(),
                    'optimizer_actor_state_dict': optimizer_actor.state_dict(),
                    'optimizer_critic_state_dict': optimizer_critic.state_dict(),
                    'temperature': temperature,
                    'best_reward': best_reward,
                    'global_step': global_step
                }, os.path.join(checkpoints_dir, "best_model.pth"))
                print(f"에포크 {epoch}: 최고 보상 갱신: {best_reward:.2f}")
            else:
                no_improvement_count += 1
                print(f"에포크 {epoch}: 보상 {val_reward:.2f}, 개선 없음: {no_improvement_count}")

        if epoch % config.checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_actor_state_dict': optimizer_actor.state_dict(),
                'optimizer_critic_state_dict': optimizer_critic.state_dict(),
                'temperature': temperature,
                'best_reward': best_reward,
                'global_step': global_step
            }, os.path.join(checkpoints_dir, f"model_epoch_{epoch}.pth"))

        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        avg_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        wandb.log({
            "epoch": epoch,
            "loss": avg_loss,
            "reward": avg_reward,
            "temperature": temperature,
            "learning_rate_actor": scheduler_actor.get_last_lr()[0],
            "learning_rate_critic": scheduler_critic.get_last_lr()[0]
        })

        if no_improvement_count >= config.early_stopping_patience:
            print(f"조기 중단: {config.early_stopping_patience} 에포크 동안 개선 없음")
            break

    wandb.finish()

# ### 메인 함수
def main():
    parser = argparse.ArgumentParser(description="다중 UAV 미션 할당 학습")
    parser.add_argument('--num_uavs', type=int, default=3, help='UAV 수')
    parser.add_argument('--num_missions', type=int, default=20, help='미션 수')
    parser.add_argument('--num_epochs', type=int, default=100, help='학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--lr_actor', type=float, default=1e-4, help='액터 학습률')
    parser.add_argument('--lr_critic', type=float, default=1e-4, help='크리틱 학습률')
    parser.add_argument('--temperature', type=float, default=1.0, help='샘플링 온도')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='체크포인트 경로')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    data = MissionData(args.num_missions, args.num_uavs, seed=args.seed, device=device)
    config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        temperature=args.temperature
    )

    train_env = MissionEnvironment(
        data.missions, data.uavs_start, data.uavs_end, data.uavs_speeds, data.uav_types,
        data.risk_centers, data.risk_radii, data.zone_centers, data.zone_radii,
        device, seed=args.seed, curriculum_epoch=0, total_epochs=args.num_epochs
    )
    val_env = MissionEnvironment(
        data.missions, data.uavs_start, data.uavs_end, data.uavs_speeds, data.uav_types,
        data.risk_centers, data.risk_radii, data.zone_centers, data.zone_radii,
        device, seed=args.seed + 1, curriculum_epoch=0, total_epochs=args.num_epochs
    )

    policy_net = TransformerActorCriticNetwork(args.num_missions, args.num_uavs).to(device)
    optimizer_actor = optim.Adam(policy_net.actor.parameters(), lr=args.lr_actor)
    optimizer_critic = optim.Adam(policy_net.critic.parameters(), lr=args.lr_critic)
    scheduler_actor = WarmupScheduler(optimizer_actor, config.warmup_steps, optim.lr_scheduler.StepLR(optimizer_actor, step_size=10, gamma=0.9))
    scheduler_critic = WarmupScheduler(optimizer_critic, config.warmup_steps, optim.lr_scheduler.StepLR(optimizer_critic, step_size=10, gamma=0.9))
    edge_indices_cache = precompute_edge_indices(args.num_missions, device)

    train_model(train_env, val_env, policy_net, optimizer_actor, optimizer_critic, scheduler_actor, scheduler_critic,
                device, edge_indices_cache, config, args.checkpoint_path)

if __name__ == "__main__":
    main()