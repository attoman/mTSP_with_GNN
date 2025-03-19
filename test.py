import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm
import wandb
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional, Union

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

# ============================
# 설정 클래스
# ============================

@dataclass
class TrainingConfig:
    """학습 매개변수를 관리하는 통합 설정 클래스"""
    # 기본 학습 매개변수
    num_epochs: int
    batch_size: int
    lr_actor: float
    lr_critic: float
    
    # 액션 샘플링 관련 매개변수
    temperature: float
    temperature_min: float = 0.1
    temperature_decay: float = 0.995
    
    # 보상 계산 관련 매개변수
    reward_type: str = 'mixed'
    alpha: float = 0.5  # 총 시간에 대한 가중치
    beta: float = 0.3   # 시간 표준편차에 대한 가중치
    gamma: float = 0.1  # 최대 시간에 대한 가중치
    
    # 체크포인트 및 검증 관련 매개변수
    checkpoint_interval: int = 10  # N 에포크마다 체크포인트 저장
    validation_interval: int = 5   # N 에포크마다 검증
    
    # 학습 안정화 매개변수
    gradient_clip: float = 1.0     # 그래디언트 클리핑 값
    early_stopping_patience: int = 15  # 조기 중단 인내
    
    # 커리큘럼 학습 매개변수
    use_curriculum: bool = True    # 커리큘럼 학습 활성화
    curriculum_min_missions: int = 5  # 커리큘럼 시작 미션 수
    
    # 모델 구조 매개변수
    hidden_dim: int = 128
    num_gat_layers: int = 3
    gat_heads: int = 8
    dropout: float = 0.1
    
    # 환경 설정
    max_step_limit: int = 100  # 무한 루프 방지용 최대 단계 제한
    risk_penalty: float = 10.0  # 위험 구역 통과 시 페널티

# ============================
# 유틸리티 함수
# ============================

def calculate_distance(mission1: torch.Tensor, mission2: torch.Tensor) -> torch.Tensor:
    """두 미션 간의 유클리드 거리를 계산합니다."""
    return torch.sqrt(torch.sum((mission1 - mission2) ** 2) + 1e-8)

def calculate_distance_matrix(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
    """두 점 집합 간의 거리 행렬을 벡터화된 방식으로 계산합니다."""
    p1 = points1.unsqueeze(1)  # [N, 1, 2]
    p2 = points2.unsqueeze(0)  # [1, M, 2]
    return torch.sqrt(torch.sum((p1 - p2) ** 2, dim=2) + 1e-8)

def compute_segment_circle_intersections(positions: torch.Tensor, targets: torch.Tensor, 
                                        circle_centers: torch.Tensor, circle_radii: torch.Tensor) -> torch.Tensor:
    """선분-원 교차를 벡터화된 방식으로 계산합니다."""
    n_positions = positions.shape[0]
    n_targets = targets.shape[0]
    n_circles = circle_centers.shape[0] if len(circle_centers.shape) > 0 else 0
    device = positions.device
    
    # 원이 없으면 교차도 없음
    if n_circles == 0:
        return torch.zeros((n_positions, n_targets), dtype=torch.bool, device=device)
        
    intersections = torch.zeros((n_positions, n_targets, n_circles), dtype=torch.bool, device=device)
    
    # 각 원에 대해 교차 검사 수행
    for c_idx in range(n_circles):
        center = circle_centers[c_idx]
        radius = circle_radii[c_idx]
        
        # 브로드캐스팅을 위한 텐서 준비
        A = positions.unsqueeze(1)  # [N, 1, 2]
        B = targets.unsqueeze(0)    # [1, M, 2]
        C = center.unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
        
        # 벡터 계산
        d = B - A  # [N, M, 2]
        f = A - C  # [N, M, 2]
        
        # 2차 방정식 계수 계산
        a = torch.sum(d * d, dim=2)  # [N, M]
        b = 2 * torch.sum(f * d, dim=2)  # [N, M]
        c = torch.sum(f * f, dim=2) - radius**2  # [N, M]
        
        # 판별식 계산
        discriminant = b**2 - 4*a*c  # [N, M]
        valid_discriminant = discriminant >= 0
        
        # t 값 계산
        t1 = torch.zeros_like(a)
        t2 = torch.zeros_like(a)
        
        # 0으로 나누기 방지
        valid_a = a != 0
        mask = valid_discriminant & valid_a
        
        if mask.any():
            sqrt_disc = torch.sqrt(discriminant[mask])
            t1[mask] = (-b[mask] - sqrt_disc) / (2 * a[mask])
            t2[mask] = (-b[mask] + sqrt_disc) / (2 * a[mask])
        
        # 선분 위 교차점 검사
        t1_valid = (0 <= t1) & (t1 <= 1)
        t2_valid = (0 <= t2) & (t2 <= 1)
        
        # 결과 저장
        intersections[:, :, c_idx] = t1_valid | t2_valid
    
    # 어떤 원과라도 교차하는지 검사 ([N, M] 형태로 축소)
    return intersections.any(dim=-1)

def create_edge_index(num_missions: int) -> torch.Tensor:
    """그래프 엣지 인덱스 생성"""
    adj = torch.ones((num_missions, num_missions))
    edge_index, _ = dense_to_sparse(adj)
    return edge_index

def precompute_edge_indices(max_missions: int, device: torch.device) -> Dict[int, torch.Tensor]:
    """미션 수에 따른 엣지 인덱스를 미리 계산하여 캐싱"""
    cache = {}
    for n in range(2, max_missions + 1):  # 최소 2개(시작/종료) 부터 최대 미션 수까지
        cache[n] = create_edge_index(n).to(device)
    return cache

# ============================
# 데이터 및 환경 클래스
# ============================

class MissionData:
    """미션 데이터 생성 및 관리 클래스"""
    def __init__(self, num_missions: int = 20, num_uavs: int = 3, 
                 seed: Optional[int] = None, device: torch.device = torch.device('cpu')):
        """
        Args:
            num_missions: 총 미션 수 (시작 및 종료 지점 포함)
            num_uavs: UAV 수
            seed: 랜덤 시드
            device: 텐서를 저장할 장치
        """
        self.num_missions = num_missions
        self.num_uavs = num_uavs
        self.seed = seed
        self.device = device
        
        # 랜덤 시드 설정
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
        
        # 데이터 생성
        self.missions, self.uavs_start, self.uavs_end, self.uavs_speeds = self._generate_mission_data()
        self.risk_areas = self._generate_risk_areas()
        self.no_entry_zones = self._generate_no_entry_zones()
        
        # 벡터화된 계산을 위한 텐서 생성
        self.risk_centers, self.risk_radii = self._create_area_tensors(self.risk_areas)
        self.zone_centers, self.zone_radii = self._create_area_tensors(self.no_entry_zones)

    def _generate_mission_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """랜덤 미션 위치 및 UAV 매개변수 생성"""
        # 미션 좌표 생성 (시작 및 종료 지점 별도 설정)
        missions = torch.rand((self.num_missions - 2, 2), device=self.device) * 100  # 중간 미션들
        start_point = torch.rand((1, 2), device=self.device) * 100  # 시작 지점
        end_point = torch.rand((1, 2), device=self.device) * 100    # 도착 지점 (별도)
        
        # 시작점, 중간 미션들, 도착점 순으로 미션 배열 구성
        missions = torch.cat([start_point, missions, end_point], dim=0)
        
        # 모든 UAV는 시작 지점에서 출발
        uavs_start = start_point.repeat(self.num_uavs, 1)
        uavs_end = end_point.repeat(self.num_uavs, 1)
        
        # UAV 속도 설정 (5~30 사이 랜덤)
        uavs_speeds = torch.randint(5, 30, (self.num_uavs,), dtype=torch.float, device=self.device)
        
        return missions, uavs_start, uavs_end, uavs_speeds

    def _generate_risk_areas(self) -> List[Dict[str, Union[torch.Tensor, float]]]:
        """위험 지역 랜덤 생성 (통과 가능하지만 비용 증가)"""
        num_risk_areas = random.randint(1, 5)
        return [{'center': torch.rand(2, device=self.device) * 100, 
                'radius': random.uniform(5, 15)} 
                for _ in range(num_risk_areas)]

    def _generate_no_entry_zones(self) -> List[Dict[str, Union[torch.Tensor, float]]]:
        """출입 불가 지역 랜덤 생성 (완전히 통과 불가)"""
        num_no_entry_zones = random.randint(1, 3)
        return [{'center': torch.rand(2, device=self.device) * 100, 
                'radius': random.uniform(3, 10)} 
                for _ in range(num_no_entry_zones)]

    def _create_area_tensors(self, areas: List[Dict[str, Union[torch.Tensor, float]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """영역 목록에서 중심점과 반지름 텐서 생성"""
        if not areas:  # 영역이 없는 경우 빈 텐서 반환
            return torch.empty((0, 2), device=self.device), torch.empty(0, device=self.device)
            
        centers = torch.stack([area['center'] for area in areas])
        radii = torch.tensor([area['radius'] for area in areas], device=self.device)
        return centers, radii

class MissionEnvironment:
    """다중 UAV 미션 할당 환경 클래스"""
    def __init__(self, missions: torch.Tensor, uavs_start: torch.Tensor, uavs_end: torch.Tensor, 
                 uavs_speeds: torch.Tensor, risk_centers: torch.Tensor, risk_radii: torch.Tensor, 
                 zone_centers: torch.Tensor, zone_radii: torch.Tensor, 
                 device: torch.device = torch.device('cpu'), seed: Optional[int] = None, 
                 curriculum_epoch: Optional[int] = None, total_epochs: Optional[int] = None,
                 min_missions: int = 5):
        """
        Args:
            missions: 모든 미션 좌표 ([N, 2] 텐서)
            uavs_start: UAV 시작 위치 ([U, 2] 텐서)
            uavs_end: UAV 도착 위치 ([U, 2] 텐서)
            uavs_speeds: UAV 속도 ([U] 텐서)
            risk_centers: 위험 지역 중심 좌표 ([R, 2] 텐서)
            risk_radii: 위험 지역 반지름 ([R] 텐서)
            zone_centers: 출입 금지 구역 중심 좌표 ([Z, 2] 텐서)
            zone_radii: 출입 금지 구역 반지름 ([Z] 텐서)
            device: 사용할 장치
            seed: 랜덤 시드
            curriculum_epoch: 현재 커리큘럼 에포크
            total_epochs: 총 에포크 수
            min_missions: 커리큘럼 시작 시 최소 미션 수
        """
        self.device = device
        self.seed = seed
        
        # 랜덤 시드 설정
        if self.seed is not None:
            torch.manual_seed(self.seed)
            random.seed(self.seed)
        
        # 환경 매개변수
        self.original_missions = missions.clone()
        self.max_missions = missions.size(0)
        self.uavs_start = uavs_start
        self.uavs_end = uavs_end
        self.speeds = uavs_speeds
        
        # 위험/금지 구역 정보
        self.risk_centers = risk_centers
        self.risk_radii = risk_radii
        self.zone_centers = zone_centers
        self.zone_radii = zone_radii
        
        # 커리큘럼 학습 매개변수
        self.curriculum_epoch = curriculum_epoch
        self.total_epochs = total_epochs
        self.use_curriculum = curriculum_epoch is not None and total_epochs is not None
        self.min_missions = min_missions
        
        # 초기화
        self.reset()

    def adjust_curriculum(self) -> int:
        """커리큘럼에 따라 미션 수를 점진적으로 조절"""
        if not self.use_curriculum:
            return self.max_missions
            
        # 학습 진행에 따라 미션 수 조절 (부드러운 진행)
        progress = min(1.0, self.curriculum_epoch / (self.total_epochs * 0.8))
        num_missions = int(self.min_missions + (self.max_missions - self.min_missions) * progress)
        return min(max(self.min_missions, num_missions), self.max_missions)

    def reset(self) -> Dict[str, torch.Tensor]:
        """환경을 초기 상태로 되돌리고 초기 상태 반환"""
        # 커리큘럼 모드에서는 미션 부분집합 선택
        if self.use_curriculum:
            num_missions = self.adjust_curriculum()
            
            # 시작점과 종료점은 항상 포함
            selected_indices = [0]  # 시작점
            
            # 중간 미션 수가 0보다 크면 랜덤 선택
            middle_count = num_missions - 2
            if middle_count > 0:
                middle_indices = random.sample(range(1, self.max_missions - 1), min(middle_count, self.max_missions - 2))
                selected_indices.extend(sorted(middle_indices))
            
            # 종료점 추가
            selected_indices.append(self.max_missions - 1)
            
            # 선택된 미션 추출
            self.missions = self.original_missions[selected_indices].clone()
        else:
            # 커리큘럼 아닌 경우 전체 미션 사용
            self.missions = self.original_missions.clone()
        
        # 미션 및 UAV 수 업데이트
        self.num_missions = self.missions.size(0)
        self.num_uavs = self.uavs_start.size(0)
        
        # UAV 상태 초기화
        self.current_positions = self.uavs_start.clone()
        self.visited = torch.zeros(self.num_missions, dtype=torch.bool, device=self.device)
        self.visited[0] = True  # 시작점은 이미 방문한 것으로 표시
        self.reserved = torch.zeros_like(self.visited)
        self.paths = [[0] for _ in range(self.num_uavs)]  # 각 UAV 경로 초기화
        self.cumulative_travel_times = torch.zeros(self.num_uavs, device=self.device)
        self.ready_for_next_action = torch.ones(self.num_uavs, dtype=torch.bool, device=self.device)
        self.targets = torch.full((self.num_uavs,), -1, dtype=torch.long, device=self.device)
        self.remaining_distances = torch.full((self.num_uavs,), float('inf'), device=self.device)
        
        return self.get_state()

    def get_state(self) -> Dict[str, torch.Tensor]:
        """현재 환경 상태 반환"""
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
        """각 UAV에 대한 액션 마스크 생성 (True = 해당 액션 금지)"""
        # 기본 마스크: 이미 방문했거나 예약된 미션은 선택 불가
        mask_base = (self.visited | self.reserved).unsqueeze(0).repeat(self.num_uavs, 1)
        
        # 기본적으로 도착 지점은 마스킹
        mask_base[:, -1] = True
        
        # 모든 중간 미션이 완료되면 도착 지점으로 이동 허용
        if self.visited[1:-1].all():
            mask_base[:, -1] = False
        
        # 준비되지 않은 UAV는 모든 액션 마스킹
        mask_base[~self.ready_for_next_action] = True
        
        # 출입 금지 구역 검사 (벡터화된 계산)
        if self.zone_centers.shape[0] > 0:  # 출입 금지 구역이 있는 경우에만
            # 준비된 UAV만 마스크 업데이트
            ready_uavs = torch.where(self.ready_for_next_action)[0]
            if len(ready_uavs) > 0:
                ready_positions = self.current_positions[ready_uavs]
                
                # 각 준비된 UAV에 대해 모든 미션과의 교차 검사
                intersections = compute_segment_circle_intersections(
                    ready_positions, self.missions, 
                    self.zone_centers, self.zone_radii
                )
                
                # 교차하는 미션 마스킹
                for i, u_idx in enumerate(ready_uavs):
                    mask_base[u_idx] |= intersections[i]
        
        return mask_base

    def calculate_cost_matrix(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """미션 할당을 위한 비용 행렬 계산"""
        # 모든 UAV-미션 쌍에 대한 거리 행렬 계산
        dist_matrix = calculate_distance_matrix(self.current_positions, self.missions)
        
        # 시간 행렬 계산 (거리/속도)
        speeds_expanded = self.speeds.unsqueeze(1)  # [num_uavs, 1]
        timetogo_matrix = dist_matrix / (speeds_expanded + 1e-5)
        
        # 위험 구역 페널티 계산 (있는 경우)
        if self.risk_centers.shape[0] > 0:
            # 모든 UAV와 미션 사이 선분이 위험 구역과 교차하는지 검사
            intersections = compute_segment_circle_intersections(
                self.current_positions, self.missions,
                self.risk_centers, self.risk_radii
            )
            
            # 교차하는 경우 페널티 추가
            timetogo_matrix += intersections.float() * 10.0
        
        return timetogo_matrix, dist_matrix

    def step(self, actions: List[int]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, bool]:
        """환경에서 액션 실행 및 다음 상태 반환"""
        actions = torch.tensor(actions, device=self.device)
        travel_times = torch.zeros(self.num_uavs, device=self.device)
        
        # 각 UAV에 미션 할당
        for u, action in enumerate(actions):
            if not self.ready_for_next_action[u] or action == -1:
                continue
                
            if not self.visited[action] and not self.reserved[action]:
                self.reserved[action] = True
                self.ready_for_next_action[u] = False
                self.targets[u] = action
                self.remaining_distances[u] = calculate_distance(
                    self.current_positions[u], self.missions[action]
                )
        
        # 각 UAV를 목표 지점으로 이동
        for u, target in enumerate(self.targets):
            if target != -1 and not self.ready_for_next_action[u]:
                # 이동 시간 계산
                travel_time = self.remaining_distances[u] / (self.speeds[u] + 1e-5)
                
                # UAV 상태 업데이트
                self.cumulative_travel_times[u] += travel_time
                self.current_positions[u] = self.missions[target]
                self.visited[target] = True
                self.paths[u].append(target.item())
                self.ready_for_next_action[u] = True
                self.reserved[target] = False
                self.targets[u] = -1
                travel_times[u] = travel_time
        
        # 모든 중간 미션이 완료되었는지 확인
        all_mid_missions_done = self.visited[1:-1].all().item()
        
        # 중간 미션이 모두 완료되면 남은 UAV를 도착 지점으로 유도
        if all_mid_missions_done:
            for u in range(self.num_uavs):
                if self.ready_for_next_action[u] and self.targets[u] != self.num_missions - 1:
                    self.targets[u] = self.num_missions - 1
                    self.ready_for_next_action[u] = False
                    self.remaining_distances[u] = calculate_distance(
                        self.current_positions[u], self.missions[-1]
                    )
        
        # 에피소드 종료 여부 확인
        done = all_mid_missions_done and self.visited[-1].item()
        
        return self.get_state(), travel_times, done

# ============================
# 신경망 정의
# ============================

class TypeAwareGraphAttention(nn.Module):
    """타입 인식 그래프 어텐션 레이어"""
    def __init__(self, in_features: int, out_features: int, heads: int = 8, 
                dropout: float = 0.1, types: int = 3):
        super(TypeAwareGraphAttention, self).__init__()
        self.types = types
        
        # 각 노드 유형에 대한 투영 레이어
        self.type_projections = nn.ModuleList([
            nn.Linear(in_features, in_features) for _ in range(types)
        ])
        
        # GAT 레이어
        self.gat = GATConv(in_features, out_features // heads, 
                          heads=heads, dropout=dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                node_types: torch.Tensor) -> torch.Tensor:
        """순방향 전파"""
        # 노드 유형별 투영
        x_projected = torch.zeros_like(x)
        for type_id in range(self.types):
            mask = (node_types == type_id)
            if mask.sum() > 0:
                x_projected[mask] = self.type_projections[type_id](x[mask])
        
        # GAT 적용
        return self.gat(x_projected, edge_index)

class AllocationNetwork(nn.Module):
    """미션 할당 결정을 위한 액터 네트워크"""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, 
                num_layers: int = 3, heads: int = 8, dropout: float = 0.1):
        super(AllocationNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 초기 특성 투영
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 그래프 어텐션 레이어
        self.tga_layers = nn.ModuleList([
            TypeAwareGraphAttention(hidden_dim, hidden_dim, heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # 레이어 정규화
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # 최종 MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                node_types: torch.Tensor, batch=None) -> torch.Tensor:
        """순방향 전파"""
        # 초기 투영
        x = self.input_proj(x)
        
        # 각 GAT 레이어 적용
        for layer, norm in zip(self.tga_layers, self.layer_norms):
            # 잔차 연결을 위해 입력 저장
            x_res = x
            
            # GAT 레이어
            x = layer(x, edge_index, node_types)
            x = F.relu(x)
            
            # 잔차 연결 및 정규화
            x = norm(x + x_res)
            
        # 최종 점수 예측
        return self.mlp(x)

class SurrogateNetwork(nn.Module):
    """가치 추정을 위한 크리틱 네트워크"""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, 
                num_layers: int = 3, heads: int = 8, dropout: float = 0.1):
        super(SurrogateNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 초기 특성 투영
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 그래프 어텐션 레이어
        self.embedding_layers = nn.ModuleList([
            TypeAwareGraphAttention(hidden_dim, hidden_dim, heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # 레이어 정규화
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # 비용 예측 MLP
        self.cost_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                node_types: torch.Tensor, batch=None) -> torch.Tensor:
        """순방향 전파"""
        # 초기 투영
        x = self.input_proj(x)
        
        # 각 GAT 레이어 적용
        for layer, norm in zip(self.embedding_layers, self.layer_norms):
            # 잔차 연결을 위해 입력 저장
            x_res = x
            
            # GAT 레이어
            x = layer(x, edge_index, node_types)
            x = F.relu(x)
            
            # 잔차 연결 및 정규화
            x = norm(x + x_res)
            
        # 그래프 풀링 (배치가 있는 경우)
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, batch)
            
        # 비용 예측
        return self.cost_predictor(x)

class ActorCriticNetwork(nn.Module):
    """액터-크리틱 통합 네트워크"""
    def __init__(self, max_missions: int, num_uavs: int, hidden_dim: int = 128, 
                num_layers: int = 3, heads: int = 8, dropout: float = 0.1):
        super(ActorCriticNetwork, self).__init__()
        self.max_missions = max_missions
        self.num_uavs = num_uavs
        
        # 액터와 크리틱 네트워크
        self.actor = AllocationNetwork(
            input_dim=2, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers,
            heads=heads,
            dropout=dropout
        )
        
        self.critic = SurrogateNetwork(
            input_dim=2, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers,
            heads=heads,
            dropout=dropout
        )

    def forward(self, missions: torch.Tensor, edge_index: torch.Tensor, 
                batch, uav_positions: torch.Tensor, action_mask: torch.Tensor, 
                speeds: torch.Tensor, dist_matrix: torch.Tensor, 
                timetogo_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """순방향 전파"""
        # 현재 미션 수 가져오기
        num_missions = missions.size(0)
        
        # 노드 특성 및 유형 준비
        node_features = missions
        node_types = torch.zeros(num_missions, dtype=torch.long, device=missions.device)
        
        # 액터에서 액션 로짓 가져오기
        action_logits = self.actor(node_features, edge_index, node_types, batch)
        action_logits = action_logits.view(self.num_uavs, num_missions)
        
        # 액션 마스크 적용 (불가능한 액션에 -inf)
        action_logits[action_mask] = -float('inf')
        
        # 크리틱에서 상태 가치 가져오기
        state_value = self.critic(node_features, edge_index, node_types, batch)
        
        return action_logits, state_value

# ============================
# 학습 보조 함수
# ============================

def choose_action(action_logits: torch.Tensor, temperature: float, 
                 uav_order: torch.Tensor, action_mask: torch.Tensor) -> List[int]:
    """각 UAV에 대한 액션 샘플링

    Args:
        action_logits: UAV별 미션 로짓 ([num_uavs, num_missions] 텐서)
        temperature: 샘플링 온도 (높을수록 더 탐색적)
        uav_order: UAV 실행 순서
        action_mask: 액션 마스크 (True = 불가능한 액션)

    Returns:
        각 UAV에 대한 선택된 미션 인덱스 목록
    """
    # 온도 스케일링 및 소프트맥스로 확률 계산
    probs = F.softmax(action_logits / temperature, dim=-1)
    
    actions = []
    for i in range(len(uav_order)):
        uav_idx = uav_order[i].item()
        
        # 모든 액션이 마스킹된 경우
        if action_mask[uav_idx].all():
            actions.append(-1)  # 액션 없음
        else:
            # 확률 분포에서 미션 샘플링
            action = torch.multinomial(probs[uav_idx], 1).item()
            actions.append(action)
    
    return actions

def compute_episode_reward(env: MissionEnvironment, config: TrainingConfig) -> torch.Tensor:
    """에피소드 보상 계산

    Args:
        env: 미션 환경
        config: 학습 설정

    Returns:
        계산된 보상 값
    """
    # 각 UAV의 이동 시간
    travel_times = env.cumulative_travel_times
    
    if config.reward_type == 'mixed':
        # 복합 보상 계산
        total_time = travel_times.sum()
        
        # 표준 편차 (1개 이상의 UAV가 있을 때만)
        time_std = travel_times.std() if travel_times.size(0) > 1 else torch.tensor(0.0, device=env.device)
        
        # 최대 시간
        max_time = travel_times.max()
        
        # 가중치를 적용한 복합 보상
        return -(config.alpha * total_time + config.beta * time_std + config.gamma * max_time)
    else:
        # 기본 보상: 총 시간 최소화
        return -travel_times.sum()

def visualize_results(env: MissionEnvironment, save_path: str, 
                     reward: Optional[float] = None, max_images_per_run: int = 100) -> None:
    """미션 경로 시각화 및 저장

    Args:
        env: 미션 환경
        save_path: 이미지 저장 경로
        reward: 에피소드 보상 (있는 경우)
        max_images_per_run: 실행당 최대 저장 이미지 수
    """
    # 이미지 수 제한을 위한 확인
    save_dir = os.path.dirname(save_path)
    if os.path.exists(save_dir):
        existing_files = len([f for f in os.listdir(save_dir) if f.endswith('.png')])
        if existing_files >= max_images_per_run:
            return
    
    # 그림 생성
    plt.figure(figsize=(10, 10))
    
    # 미션 배치
    missions = env.missions.cpu().numpy()
    plt.scatter(missions[:, 0], missions[:, 1], c='blue', label='미션')
    plt.scatter(missions[0, 0], missions[0, 1], c='green', s=100, label='시작 지점')
    plt.scatter(missions[-1, 0], missions[-1, 1], c='red', s=100, label='도착 지점')
    
    # UAV 경로
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, path in enumerate(env.paths):
        if path:
            path_coords = missions[path]
            plt.plot(path_coords[:, 0], path_coords[:, 1], c=colors[i % len(colors)], 
                    label=f'UAV {i} (속도: {env.speeds[i].item():.1f})')
    
    # 위험 지역 표시
    for i in range(env.risk_centers.shape[0]):
        center = env.risk_centers[i].cpu().numpy()
        radius = env.risk_radii[i].item()
        circle = plt.Circle(center, radius, color='orange', alpha=0.3)
        plt.gca().add_patch(circle)
    
    # 출입 금지 구역 표시
    for i in range(env.zone_centers.shape[0]):
        center = env.zone_centers[i].cpu().numpy()
        radius = env.zone_radii[i].item()
        circle = plt.Circle(center, radius, color='red', alpha=0.5)
        plt.gca().add_patch(circle)
    
    # 제목 설정
    title = f"보상: {reward:.2f}" if reward is not None else "UAV 경로"
    if env.use_curriculum:
        title += f" | 커리큘럼 단계: {env.curriculum_epoch}/{env.total_epochs}"
    
    plt.title(title)
    plt.legend()
    
    # 이미지 저장
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def validate_model(env: MissionEnvironment, policy_net: ActorCriticNetwork, 
                  device: torch.device, edge_indices_cache: Dict[int, torch.Tensor], 
                  save_dir: str, epoch: int, config: TrainingConfig) -> float:
    """검증 데이터로 모델 평가

    Args:
        env: 검증 환경
        policy_net: 정책 네트워크
        device: 장치
        edge_indices_cache: 엣지 인덱스 캐시
        save_dir: 결과 저장 디렉토리
        epoch: 현재 에포크
        config: 학습 설정

    Returns:
        검증 보상
    """
    # 평가 모드로 전환
    policy_net.eval()
    
    # 환경 초기화
    state = env.reset()
    total_reward = 0
    
    # 현재 미션 수에 맞는 엣지 인덱스 가져오기
    num_missions = env.num_missions
    if num_missions not in edge_indices_cache:
        edge_indices_cache[num_missions] = create_edge_index(num_missions).to(device)
    edge_index = edge_indices_cache[num_missions]
    
    # 배치 텐서 생성
    batch = torch.zeros(num_missions, dtype=torch.long, device=device)
    
    # 그라디언트 계산 없이 실행
    with torch.no_grad():
        done = False
        step_count = 0
        
        # 한 에피소드 실행
        while not done and step_count < config.max_step_limit:
            step_count += 1
            
            # 액션 마스크 및 비용 행렬 계산
            action_mask = env.create_action_mask()
            timetogo_matrix, dist_matrix = env.calculate_cost_matrix()
            
            # 정책 네트워크에서 액션 로짓 가져오기
            action_logits, _ = policy_net(
                env.missions, edge_index, batch, 
                state['positions'], action_mask, 
                env.speeds, dist_matrix, timetogo_matrix
            )
            
            # 액션 선택 (낮은 온도로 거의 탐욕적)
            actions = choose_action(
                action_logits, 0.01, 
                torch.arange(env.num_uavs, device=device), action_mask
            )
            
            # 액션 실행 및 다음 상태 가져오기
            state, _, done = env.step(actions)
            
            # 보상 누적
            reward = compute_episode_reward(env, config)
            total_reward += reward
    
    # 주기적으로 시각화 저장
    if epoch % 10 == 0 or epoch == 1:
        visualize_results(env, os.path.join(save_dir, f"val_epoch_{epoch}.png"), reward=total_reward)
    
    # 학습 모드로 복귀
    policy_net.train()
    return total_reward

# ============================
# 학습 메인 함수
# ============================

def train_model(env: MissionEnvironment, val_env: MissionEnvironment, 
               policy_net: ActorCriticNetwork, optimizer_actor: optim.Optimizer, 
               optimizer_critic: optim.Optimizer, scheduler_actor, scheduler_critic, 
               device: torch.device, edge_indices_cache: Dict[int, torch.Tensor], 
               config: TrainingConfig, checkpoint_path: Optional[str] = None, 
               results_dir: str = "./results", checkpoints_dir: str = "./checkpoints") -> None:
    """모델 학습 실행

    Args:
        env: 학습 환경
        val_env: 검증 환경
        policy_net: 정책 네트워크
        optimizer_actor: 액터 옵티마이저
        optimizer_critic: 크리틱 옵티마이저
        scheduler_actor: 액터 학습률 스케줄러
        scheduler_critic: 크리틱 학습률 스케줄러
        device: 연산 장치
        edge_indices_cache: 엣지 인덱스 캐시
        config: 학습 설정
        checkpoint_path: 체크포인트 경로 (있는 경우)
        results_dir: 결과 저장 디렉토리
        checkpoints_dir: 체크포인트 저장 디렉토리
    """
    # 고유 실행 이름 생성
    run_name = f"multi_uav_mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # WandB 초기화
    wandb.init(project="multi_uav_mission", name=run_name)
    
    try:
        # 디렉토리 생성
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # 온도 초기화
        temperature = config.temperature
        best_reward = -float('inf')
        no_improvement_count = 0  # 조기 중단용 카운터
        
        # 체크포인트 로드 (있는 경우)
        start_epoch = 1
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # 모델 가중치 로드
                if isinstance(policy_net, nn.DataParallel):
                    policy_net.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    policy_net.load_state_dict(checkpoint['model_state_dict'])
                
                # 옵티마이저 상태 로드
                optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
                optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
                
                # 기타 학습 상태 로드
                start_epoch = checkpoint.get('epoch', 0) + 1
                temperature = checkpoint.get('temperature', temperature)
                best_reward = checkpoint.get('best_reward', -float('inf'))
                
                print(f"체크포인트 로드: {checkpoint_path}")
            except Exception as e:
                print(f"체크포인트 로드 오류: {e}")
        
        # 에포크 루프
        for epoch in tqdm(range(start_epoch, config.num_epochs + 1), desc="에포크"):
            # 커리큘럼 에포크 업데이트
            if config.use_curriculum:
                env.curriculum_epoch = epoch
                val_env.curriculum_epoch = epoch
            
            # 에포크 메트릭 초기화
            epoch_losses = []
            epoch_rewards = []
            
            # 미니배치 루프
            for _ in tqdm(range(config.batch_size), desc=f"에포크 {epoch}", leave=False):
                # 환경 초기화
                state = env.reset()
                done = False
                log_probs, values, rewards = [], [], []
                
                # 현재 미션 수에 대한 엣지 인덱스 가져오기
                num_missions = env.num_missions
                if num_missions not in edge_indices_cache:
                    edge_indices_cache[num_missions] = create_edge_index(num_missions).to(device)
                edge_index = edge_indices_cache[num_missions]
                
                # 배치 텐서 생성
                batch = torch.zeros(num_missions, dtype=torch.long, device=device)
                
                # 에피소드 루프
                while not done:
                    # 액션 마스크 및 비용 행렬 계산
                    action_mask = env.create_action_mask()
                    timetogo_matrix, dist_matrix = env.calculate_cost_matrix()
                    
                    # 정책 네트워크에서 액션 로짓 및 가치 가져오기
                    action_logits, state_values = policy_net(
                        env.missions, edge_index, batch, state['positions'],
                        action_mask, env.speeds, dist_matrix, timetogo_matrix
                    )
                    
                    # UAV 순서 및 액션 선택
                    uav_order = torch.arange(env.num_uavs, device=device)
                    actions = choose_action(action_logits, temperature, uav_order, action_mask)
                    
                    # 액션에 대한 로그 확률 및 가치 저장
                    for i, action in enumerate(actions):
                        if action != -1:  # 유효한 액션인 경우만
                            probs = F.softmax(action_logits[i], dim=-1)
                            log_probs.append(torch.log(probs[action] + 1e-10))
                            values.append(state_values)
                    
                    # 환경 단계 진행
                    next_state, _, done = env.step(actions)
                    
                    # 보상 계산 및 저장
                    reward = compute_episode_reward(env, config)
                    rewards.append(reward)
                    
                    # 상태 업데이트
                    state = next_state
                
                # 에피소드에서 액션이 없었으면 스킵
                if not log_probs:
                    continue
                    
                # 총 보상 계산
                R = torch.stack(rewards).sum()
                epoch_rewards.append(R.item())
                
                # 리턴 계산
                returns = torch.tensor([R] * len(values), device=device, dtype=torch.float)
                
                # 가치 스택
                values = torch.stack(values).squeeze()
                
                # 어드밴티지 계산
                advantage = returns - values.detach()
                
                # 어드밴티지 정규화 (안정성 향상)
                if advantage.shape[0] > 1:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                
                # 정책 손실 계산
                policy_loss = torch.stack([-lp * adv for lp, adv in zip(log_probs, advantage)]).mean()
                
                # 가치 손실 계산
                value_loss = F.mse_loss(values, returns)
                
                # 총 손실
                loss = policy_loss + value_loss
                epoch_losses.append(loss.item())
                
                # 옵티마이저 단계
                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                loss.backward()
                
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(
                    policy_net.parameters() if not isinstance(policy_net, nn.DataParallel) 
                    else policy_net.module.parameters(), 
                    max_norm=config.gradient_clip
                )
                
                # 옵티마이저 단계
                optimizer_actor.step()
                optimizer_critic.step()
            
            # 온도 감소
            temperature = max(temperature * config.temperature_decay, config.temperature_min)
            
            # 학습률 스케줄러 단계
            scheduler_actor.step()
            scheduler_critic.step()
            
            # 검증 (주기적으로)
            if epoch % config.validation_interval == 0:
                val_reward = validate_model(val_env, policy_net, device, edge_indices_cache, 
                                         results_dir, epoch, config)
                
                # 최고 모델 갱신
                if val_reward > best_reward:
                    best_reward = val_reward
                    no_improvement_count = 0
                    
                    # 최고 모델 저장
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': policy_net.state_dict() if not isinstance(policy_net, nn.DataParallel) 
                                          else policy_net.module.state_dict(),
                        'optimizer_actor_state_dict': optimizer_actor.state_dict(),
                        'optimizer_critic_state_dict': optimizer_critic.state_dict(),
                        'temperature': temperature,
                        'best_reward': best_reward
                    }, os.path.join(checkpoints_dir, "best_model.pth"))
                    print(f"에포크 {epoch}: 최고 보상 갱신: {best_reward:.2f}")
                else:
                    no_improvement_count += 1
                    print(f"에포크 {epoch}: 보상 {val_reward:.2f}, 개선 없음: {no_improvement_count}/{config.early_stopping_patience}")
            else:
                val_reward = None
            
            # 정기 체크포인트 저장
            if epoch % config.checkpoint_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': policy_net.state_dict() if not isinstance(policy_net, nn.DataParallel) 
                                      else policy_net.module.state_dict(),
                    'optimizer_actor_state_dict': optimizer_actor.state_dict(),
                    'optimizer_critic_state_dict': optimizer_critic.state_dict(),
                    'temperature': temperature,
                    'best_reward': best_reward
                }, os.path.join(checkpoints_dir, f"model_epoch_{epoch}.pth"))
            
            # 에포크 메트릭 계산
            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            avg_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
            
            # 커리큘럼 정보
            curriculum_info = {}
            if config.use_curriculum:
                curriculum_info = {
                    "curriculum_epoch": env.curriculum_epoch,
                    "num_missions": env.num_missions
                }
            
            # WandB 로깅
            wandb.log({
                "epoch": epoch,
                "loss": avg_loss,
                "reward": avg_reward,
                "val_reward": val_reward,
                "temperature": temperature,
                "learning_rate_actor": scheduler_actor.get_last_lr()[0],
                "learning_rate_critic": scheduler_critic.get_last_lr()[0],
                "no_improvement_count": no_improvement_count,
                **curriculum_info
            })
            
            # 조기 중단 체크
            if no_improvement_count >= config.early_stopping_patience:
                print(f"조기 중단: {config.early_stopping_patience} 에포크 동안 개선 없음")
                break
                
    finally:
        # WandB 정리
        wandb.finish()

# ============================
# 메인 함수
# ============================

def main():
    """메인 함수: 인자 처리 및 학습 실행"""
    # 인자 파싱
    parser = argparse.ArgumentParser(description="다중 UAV 미션 할당 학습")
    
    # 환경 매개변수
    parser.add_argument('--num_uavs', type=int, default=3, help='UAV 수')
    parser.add_argument('--num_missions', type=int, default=20, help='미션 수 (시작/종료 포함)')
    
    # 학습 매개변수
    parser.add_argument('--num_epochs', type=int, default=100, help='학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--lr_actor', type=float, default=1e-4, help='액터 학습률')
    parser.add_argument('--lr_critic', type=float, default=1e-4, help='크리틱 학습률')
    parser.add_argument('--temperature', type=float, default=1.0, help='초기 샘플링 온도')
    parser.add_argument('--hidden_dim', type=int, default=128, help='은닉층 차원')
    parser.add_argument('--num_layers', type=int, default=3, help='GAT 레이어 수')
    parser.add_argument('--early_stopping', type=int, default=15, help='조기 중단 인내')
    
    # 커리큘럼 매개변수
    parser.add_argument('--no_curriculum', action='store_true', help='커리큘럼 학습 비활성화')
    parser.add_argument('--min_missions', type=int, default=5, help='커리큘럼 시작 미션 수')
    
    # 경로 및 기타 설정
    parser.add_argument('--checkpoint_path', type=str, default=None, help='체크포인트 경로')
    parser.add_argument('--results_dir', type=str, default="./results", help='결과 저장 디렉토리')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--gpu_id', type=int, default=0, help='사용할 GPU ID')
    
    args = parser.parse_args()

    # GPU 설정
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 장치: {device}")
    
    # 랜덤 시드 설정
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 데이터 생성
    data = MissionData(args.num_missions, args.num_uavs, seed=args.seed, device=device)
    
    # 설정 생성
    config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        temperature=args.temperature,
        use_curriculum=not args.no_curriculum,
        early_stopping_patience=args.early_stopping,
        hidden_dim=args.hidden_dim,
        num_gat_layers=args.num_layers,
        curriculum_min_missions=args.min_missions
    )

    # 환경 생성
    train_env = MissionEnvironment(
        data.missions, data.uavs_start, data.uavs_end, data.uavs_speeds,
        data.risk_centers, data.risk_radii, data.zone_centers, data.zone_radii,
        device, seed=args.seed,
        curriculum_epoch=0 if config.use_curriculum else None,
        total_epochs=args.num_epochs if config.use_curriculum else None,
        min_missions=config.curriculum_min_missions
    )
    
    val_env = MissionEnvironment(
        data.missions, data.uavs_start, data.uavs_end, data.uavs_speeds,
        data.risk_centers, data.risk_radii, data.zone_centers, data.zone_radii,
        device, seed=args.seed+1,
        curriculum_epoch=0 if config.use_curriculum else None,
        total_epochs=args.num_epochs if config.use_curriculum else None,
        min_missions=config.curriculum_min_missions
    )

    # 엣지 인덱스 미리 계산
    edge_indices_cache = precompute_edge_indices(args.num_missions, device)
    
    # 정책 네트워크 생성
    policy_net = ActorCriticNetwork(
        args.num_missions, args.num_uavs, 
        hidden_dim=config.hidden_dim,
        num_layers=config.num_gat_layers,
        heads=config.gat_heads,
        dropout=config.dropout
    ).to(device)
    
    # 다중 GPU 지원
    if torch.cuda.device_count() > 1:
        print(f"다중 GPU 사용: {torch.cuda.device_count()} 개")
        policy_net = nn.DataParallel(policy_net)
    
    # 옵티마이저 생성
    optimizer_actor = optim.Adam(
        policy_net.module.actor.parameters() if isinstance(policy_net, nn.DataParallel) 
        else policy_net.actor.parameters(), 
        lr=args.lr_actor
    )
    
    optimizer_critic = optim.Adam(
        policy_net.module.critic.parameters() if isinstance(policy_net, nn.DataParallel) 
        else policy_net.critic.parameters(), 
        lr=args.lr_critic
    )
    
    # 학습률 스케줄러
    scheduler_actor = optim.lr_scheduler.StepLR(optimizer_actor, step_size=10, gamma=0.9)
    scheduler_critic = optim.lr_scheduler.StepLR(optimizer_critic, step_size=10, gamma=0.9)

    # 학습 시작
    print(f"커리큘럼 학습{' 사용' if config.use_curriculum else ' 없이'} 학습 시작...")
    train_model(
        train_env, val_env, policy_net, optimizer_actor, optimizer_critic,
        scheduler_actor, scheduler_critic, device, edge_indices_cache,
        config, args.checkpoint_path, args.results_dir, 
        os.path.join(args.results_dir, "checkpoints")
    )
    print("학습 완료.")

if __name__ == "__main__":
    main()