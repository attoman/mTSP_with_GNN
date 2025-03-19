# wandb 및 tqdm 진행도 모니터링 설정 개선
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
from tqdm.auto import tqdm, trange
import wandb
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
import time
import sys
from IPython.display import clear_output

# wandb 설정 클래스 추가
@dataclass
class WandBConfig:
    """WandB 로깅 설정"""
    project: str = "multi_uav_mission"
    entity: Optional[str] = None  # wandb 사용자 또는 팀 이름
    name: Optional[str] = None    # 실행 이름 (None이면 자동 생성)
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None   # 실행에 대한 설명
    group: Optional[str] = None   # 실행 그룹
    job_type: str = "training"    # 작업 유형 (training, eval, demo)
    log_model: bool = True        # 모델 로깅 여부
    log_code: bool = True         # 코드 로깅 여부
    config_exclude_keys: List[str] = field(default_factory=lambda: ["verbose"])
    mode: str = "online"          # online, offline, disabled
    save_code: bool = True        # 코드 저장 여부
    monitor_gym: bool = False     # gym 환경 모니터링 여부

    def initialize(self, config: Any = None) -> None:
        """WandB 초기화
        
        Args:
            config: 로깅할 설정 객체
        """
        if self.mode == "disabled":
            return
        
        # 실행 이름 자동 생성 (지정되지 않은 경우)
        if self.name is None:
            self.name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # wandb 초기화
        run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            tags=self.tags,
            notes=self.notes,
            group=self.group,
            job_type=self.job_type,
            config=config,
            save_code=self.save_code,
            monitor_gym=self.monitor_gym,
            mode=self.mode
        )
        
        # 코드 로깅
        if self.log_code:
            wandb.run.log_code(".")
            
        return run

# 향상된 tqdm 프로그레스바 클래스
class EnhancedProgressBar:
    """향상된 tqdm 프로그레스바 클래스"""
    def __init__(self, total: int, desc: str = None, leave: bool = True, 
                 position: int = 0, dynamic_ncols: bool = True, 
                 unit: str = 'it', nested: bool = False, 
                 metrics: Dict[str, Any] = None, log_interval: int = 10):
        """
        Args:
            total: 총 반복 횟수
            desc: 설명
            leave: 진행 표시줄 유지 여부
            position: 표시 위치
            dynamic_ncols: 동적 열 너비 여부
            unit: 단위
            nested: 중첩 진행 표시줄 여부
            metrics: 표시할 지표 딕셔너리
            log_interval: 로깅 간격
        """
        self.metrics = {} if metrics is None else metrics
        self.log_interval = log_interval
        self.last_log_time = time.time()
        self.log_count = 0
        
        # tqdm 설정
        self.pbar = tqdm(
            total=total, 
            desc=desc,
            leave=leave,
            position=position,
            dynamic_ncols=dynamic_ncols,
            unit=unit,
            miniters=1,
            smoothing=0.1
        )
        
        # 초기 지표 표시
        self._update_postfix()
    
    def update(self, n: int = 1, metrics: Dict[str, Any] = None) -> None:
        """진행 표시줄 업데이트
        
        Args:
            n: 증가량
            metrics: 업데이트할 지표
        """
        self.pbar.update(n)
        
        # 지표 업데이트
        if metrics:
            self.metrics.update(metrics)
            self._update_postfix()
        
        # 로깅 간격 확인
        current_time = time.time()
        if current_time - self.last_log_time > self.log_interval:
            self.log_count += 1
            self.last_log_time = current_time
            self._update_postfix()
    
    def _update_postfix(self) -> None:
        """진행 표시줄 접미사 업데이트"""
        # 지표 형식화
        formatted_metrics = {}
        for k, v in self.metrics.items():
            if isinstance(v, float):
                formatted_metrics[k] = f"{v:.4f}"
            else:
                formatted_metrics[k] = str(v)
        
        self.pbar.set_postfix(formatted_metrics)
    
    def close(self) -> None:
        """진행 표시줄 닫기"""
        self.pbar.close()

# 진행도 시각화 함수
def plot_progress(metrics: Dict[str, List[float]], 
                 title: str = "Training Progress",
                 figsize: Tuple[int, int] = (12, 8)) -> None:
    """학습 진행도 실시간 시각화
    
    Args:
        metrics: 시각화할 지표
        title: 그래프 제목
        figsize: 그림 크기
    """
    plt.figure(figsize=figsize)
    for i, (key, values) in enumerate(metrics.items()):
        if not values:
            continue
        
        plt.subplot(2, 2, i+1)
        plt.plot(values)
        plt.title(key)
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # IPython 환경 확인 방법 수정
    is_notebook = False
    try:
        # IPython 모듈을 import하여 환경 확인
        import IPython
        if IPython.get_ipython() is not None:
            is_notebook = True
    except (ImportError, NameError):
        pass
    
    if is_notebook:
        # IPython이 사용 가능한 경우에만 clear_output 호출
        from IPython.display import clear_output
        clear_output(wait=True)
        plt.show()
    else:
        # 일반 Python 환경에서는 그냥 show 호출
        plt.show()

# 학습 설정 클래스 확장
@dataclass
class TrainingConfig:
    """학습 매개변수 관리 - 보다 명확한 문서화 및 기본값 설정"""
    # 학습 기본 설정
    num_epochs: int = 100                 # 학습 에포크 수
    batch_size: int = 32                  # 배치 크기
    lr_actor: float = 1e-4                # 액터 학습률
    lr_critic: float = 1e-4               # 크리틱 학습률

    # 샘플링 온도 관련 설정
    temperature: float = 1.0              # 초기 샘플링 온도
    temperature_min: float = 0.1          # 최소 샘플링 온도
    temperature_decay: float = 0.995      # 온도 감소율
    
    # 보상 계산 관련 가중치
    reward_type: str = 'mixed'            # 보상 유형 (시간, 균형, 혼합)
    alpha: float = 0.5                    # 총 시간 가중치
    beta: float = 0.3                     # 시간 표준편차 가중치
    gamma: float = 0.1                    # 최대 시간 가중치
    delta: float = 0.2                    # 미션 분배 균형 가중치
    risk_penalty: float = 10.0            # 위험 지역 페널티
    imbalance_penalty: float = 0.2        # 불균형 페널티
    
    # 체크포인트 및 검증 설정
    checkpoint_interval: int = 10         # 체크포인트 저장 간격
    validation_interval: int = 5          # 검증 간격
    
    # 옵티마이저 및 학습 안정화 관련 설정
    gradient_clip: float = 1.0            # 그래디언트 클리핑 최대값
    early_stopping_patience: int = 15     # 조기 종료 인내심
    warmup_steps: int = 1000              # 워밍업 스텝 수
    
    # 커리큘럼 학습 관련 설정 
    use_curriculum: bool = True           # 커리큘럼 학습 사용 여부
    curriculum_min_missions: int = 5      # 최소 미션 수
    curriculum_patience: int = 5          # 커리큘럼 난이도 조정 인내심
    curriculum_threshold: float = 0.7     # 난이도 상승 기준 성공률
    
    # 모델 아키텍처 관련 설정
    hidden_dim: int = 128                 # 히든 레이어 차원
    num_gat_layers: int = 3               # GAT 레이어 수
    gat_heads: int = 8                    # GAT 헤드 수
    dropout: float = 0.1                  # 드롭아웃 비율
    transformer_nhead: int = 8            # 트랜스포머 헤드 수
    transformer_num_layers: int = 3       # 트랜스포머 레이어 수
    
    # 환경 제약사항 관련 설정
    max_sequence_length: int = 100        # 최대 시퀀스 길이
    max_step_limit: int = 100             # 최대 스텝 제한
    
    # 로깅 및 디버깅 관련 설정
    verbose: bool = False                 # 상세 로깅 여부
    log_interval: int = 10                # 로깅 간격
    live_plot: bool = False               # 실시간 플롯 여부
    use_wandb: bool = True                # wandb 사용 여부
    wandb_config: WandBConfig = field(default_factory=WandBConfig)
    
    # 진행도 표시 관련 설정
    use_tqdm: bool = True                 # tqdm 사용 여부
    tqdm_position: int = 0                # tqdm 위치
    tqdm_update_interval: int = 1         # tqdm 업데이트 간격
    
    # 에포크당 성공 미션 추적 (커리큘럼에 활용)
    success_rates: List[float] = field(default_factory=list)
    progress_metrics: Dict[str, List[float]] = field(default_factory=lambda: {
        "rewards": [], "losses": [], "val_rewards": [], "success_rates": []
    })

    def __post_init__(self):
        """설정 유효성 검증"""
        assert 0 < self.temperature_min <= self.temperature, "온도 설정이 올바르지 않습니다"
        assert 0 < self.temperature_decay < 1, "온도 감소율은 0과 1 사이여야 합니다"
        assert all(w >= 0 for w in [self.alpha, self.beta, self.gamma, self.delta]), "모든 가중치는 음수가 아니어야 합니다"
        
# 유틸리티 함수 - 최적화 및 문서화 개선
def calculate_distance(mission1: torch.Tensor, mission2: torch.Tensor) -> torch.Tensor:
    """두 미션 간 유클리드 거리 계산"""
    return torch.sqrt(torch.sum((mission1 - mission2) ** 2) + 1e-8)

def calculate_distance_matrix(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
    """두 점 집합 간 거리 행렬 계산 - 벡터화된 방식으로 효율적 계산"""
    # 차원 확장을 통한 브로드캐스팅 (p1: [N, 1, D], p2: [1, M, D])
    p1 = points1.unsqueeze(1)
    p2 = points2.unsqueeze(0)
    # 벡터화된 거리 계산
    return torch.sqrt(torch.sum((p1 - p2) ** 2, dim=2) + 1e-8)

def compute_segment_circle_intersections(
        positions: torch.Tensor, 
        targets: torch.Tensor,
        circle_centers: torch.Tensor, 
        circle_radii: torch.Tensor,
        chunk_size: Optional[int] = None, 
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
    """선분-원 교차 계산 - 메모리 효율성과 계산 최적화 개선
    
    Args:
        positions: 시작점 좌표 [N, 2]
        targets: 도착점 좌표 [M, 2]
        circle_centers: 원 중심점 좌표 [C, 2]
        circle_radii: 원 반지름 [C]
        chunk_size: 메모리 효율성을 위한 청크 크기, None이면 자동 결정
        device: 계산 장치, None이면 positions의 장치 사용
    
    Returns:
        intersections: 교차 여부 불리언 행렬 [N, M]
    """
    if device is None:
        device = positions.device
    
    n_positions = positions.shape[0]
    n_targets = targets.shape[0]
    n_circles = circle_centers.shape[0] if len(circle_centers.shape) > 0 else 0
    
    # 원이 없는 경우 빠른 반환
    if n_circles == 0:
        return torch.zeros((n_positions, n_targets), dtype=torch.bool, device=device)
    
    # 동적 청크 크기 결정 (GPU 메모리 고려)
    if chunk_size is None:
        # 배치 크기 자동 조정 - 장치의 메모리 특성에 따라 조정
        is_cuda = str(device).startswith('cuda')
        # GPU는 더 큰 청크, CPU는 작은 청크
        base_size = 500 if is_cuda else 100
        # 원의 개수에 따라 청크 크기 조정
        chunk_size = max(10, base_size // max(1, n_circles // 10))
    
    intersections = torch.zeros((n_positions, n_targets), dtype=torch.bool, device=device)
    
    # 청크 단위로 처리하여 메모리 효율성 향상
    for start in range(0, n_positions, chunk_size):
        end = min(start + chunk_size, n_positions)
        
        # 현재 청크의 선분 시작점
        A_chunk = positions[start:end].unsqueeze(1)  # [chunk, 1, 2]
        
        # 선분 방향 벡터: target - start
        d_chunk = targets.unsqueeze(0) - A_chunk  # [chunk, n_targets, 2]
        
        for c_idx in range(n_circles):
            center = circle_centers[c_idx]  # [2]
            radius = circle_radii[c_idx]    # 스칼라
            
            # 원 중심점을 적절한 차원으로 변환
            C = center.view(1, 1, -1)  # [1, 1, 2]
            
            # 시작점에서 원 중심까지의 벡터
            f_chunk = A_chunk - C  # [chunk, 1, 2]
            
            # 이차방정식 계수 계산 (벡터화된 방식)
            a = (d_chunk ** 2).sum(dim=2)  # [chunk, n_targets]
            b = 2 * (f_chunk * d_chunk).sum(dim=2)  # [chunk, n_targets]
            c = (f_chunk ** 2).sum(dim=2) - radius**2  # [chunk, n_targets]
            
            # 판별식 계산
            discriminant = b**2 - 4*a*c  # [chunk, n_targets]
            
            # 교차 조건 확인
            valid_discriminant = discriminant >= 0  # 판별식 >= 0 (해가 존재)
            valid_a = a != 0  # a != 0 (선분이 점이 아님)
            mask = valid_discriminant & valid_a
            
            # 교차하는 경우에 대해서만 계산
            if mask.any():
                sqrt_disc = discriminant[mask].sqrt()
                
                # 교차점의 매개변수 t 계산 (두 해)
                t1 = (-b[mask] - sqrt_disc) / (2 * a[mask])
                t2 = (-b[mask] + sqrt_disc) / (2 * a[mask])
                
                # 선분 내부의 교차점인지 확인 (0 <= t <= 1)
                t1_valid = (0 <= t1) & (t1 <= 1)
                t2_valid = (0 <= t2) & (t2 <= 1)
                
                # 마스크를 원래 크기로 확장하여 결과 업데이트
                has_intersection = (t1_valid | t2_valid).any(dim=1).view(-1, n_targets)
                intersections[start:end] |= has_intersection
    
    return intersections

def calculate_risk_penalty(
        positions: torch.Tensor, 
        targets: torch.Tensor,
        risk_centers: torch.Tensor, 
        risk_radii: torch.Tensor,
        base_penalty: float = 10.0
    ) -> torch.Tensor:
    """위험 지역 침범에 대한 세분화된 페널티 계산
    
    Args:
        positions: 시작점 좌표 [N, 2]
        targets: 도착점 좌표 [M, 2]
        risk_centers: 위험 지역 중심점 좌표 [R, 2]
        risk_radii: 위험 지역 반지름 [R]
        base_penalty: 기본 페널티 값
    
    Returns:
        penalties: 위험 지역 페널티 행렬 [N, M]
    """
    device = positions.device
    n_positions = positions.shape[0]
    n_targets = targets.shape[0]
    
    # 위험 지역이 없는 경우 빠른 반환
    if risk_centers.shape[0] == 0:
        return torch.zeros((n_positions, n_targets), device=device)
    
    # 기본 교차 계산
    intersections = compute_segment_circle_intersections(positions, targets, risk_centers, risk_radii)
    
    # 위험 지역별 최대 반경으로 정규화하여 위험도 계산
    max_radius = risk_radii.max()
    risk_factors = risk_radii / max_radius
    
    # 페널티 초기화
    penalties = torch.zeros((n_positions, n_targets), device=device)
    
    # 교차 지점이 있는 경우에만 정밀 페널티 계산
    if intersections.any():
        # 교차하는 선분만 선택
        crossed_indices = torch.nonzero(intersections, as_tuple=True)
        p_indices, t_indices = crossed_indices
        
        for idx in range(len(p_indices)):
            p_idx, t_idx = p_indices[idx].item(), t_indices[idx].item()
            
            # 선분의 시작점과 끝점
            pos = positions[p_idx]
            tgt = targets[t_idx]
            
            # 선분 방향 벡터 및 길이
            dir_vec = tgt - pos
            dir_len = torch.norm(dir_vec)
            
            # 교차하는 위험 지역마다 페널티 계산
            total_penalty = 0.0
            for r_idx in range(risk_centers.shape[0]):
                center = risk_centers[r_idx]
                radius = risk_radii[r_idx]
                risk_factor = risk_factors[r_idx]
                
                # 선분과 원 사이의 최소 거리 계산
                # 정규화된 방향 벡터
                if dir_len > 1e-8:  # 0으로 나누기 방지
                    norm_dir = dir_vec / dir_len
                    
                    # 원 중심에서 선분에 내린 수선의 발 계산
                    proj = torch.dot(center - pos, norm_dir)
                    # 수선의 발이 선분 내에 있는지 확인
                    proj_clamped = torch.clamp(proj, 0, dir_len)
                    # 수선의 발 좌표
                    proj_point = pos + norm_dir * proj_clamped
                    # 원 중심에서 수선의 발까지의 거리
                    dist = torch.norm(center - proj_point)
                    
                    # 침범 정도에 따른 페널티 (0~1 사이 값)
                    if dist < radius:
                        # 침범 비율 계산 (0: 경계선, 1: 중심점)
                        penetration = (radius - dist) / radius
                        # 위험도와 침범 정도에 따른 페널티
                        total_penalty += base_penalty * risk_factor * penetration.item()
            
            # 총 페널티 설정
            penalties[p_idx, t_idx] = total_penalty
    
    return penalties

def create_edge_index(num_missions: int) -> torch.Tensor:
    """그래프 엣지 인덱스 생성 - 완전 연결 그래프"""
    # 모든 노드 간 연결 (자기 자신 포함)
    adj = torch.ones((num_missions, num_missions))
    # PyTorch Geometric 형식으로 변환
    edge_index, _ = dense_to_sparse(adj)
    return edge_index

def precompute_edge_indices(max_missions: int, device: torch.device) -> Dict[int, torch.Tensor]:
    """엣지 인덱스 캐싱 - 미리 계산하여 메모리 효율성 개선"""
    cache = {}
    for n in range(2, max_missions + 1):
        cache[n] = create_edge_index(n).to(device)
    return cache

def get_subsequent_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Transformer 디코더용 후속 마스크 생성
    
    시퀀스의 미래 토큰을 마스킹하는 상삼각 행렬 생성
    
    Args:
        seq_len: 시퀀스 길이
        device: 계산 장치
    
    Returns:
        mask: 상삼각 불리언 마스크 [seq_len, seq_len]
    """
    # 상삼각 행렬 생성 (대각선 제외)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask

def normalize_reward_components(
        total_time: torch.Tensor, 
        time_std: torch.Tensor, 
        max_time: torch.Tensor, 
        num_uavs: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """보상 구성 요소 정규화
    
    Args:
        total_time: 총 이동 시간
        time_std: 시간 표준편차
        max_time: 최대 이동 시간
        num_uavs: UAV 수
    
    Returns:
        norm_total: 정규화된 총 시간
        norm_std: 정규화된 표준편차
        norm_max: 정규화된 최대 시간
    """
    # UAV당 평균 시간으로 정규화
    norm_total = total_time / num_uavs
    
    # 평균 대비 표준편차로 정규화 (변동 계수)
    avg_time = total_time / num_uavs
    if avg_time > 1e-8:
        norm_std = time_std / avg_time
    else:
        norm_std = time_std
    
    # 평균 대비 최대 시간으로 정규화
    if avg_time > 1e-8:
        norm_max = max_time / avg_time - 1.0  # 평균보다 얼마나 큰지
    else:
        norm_max = max_time
        
    return norm_total, norm_std, norm_max

def compute_mission_balance_penalty(assigned_missions: List[List[int]], num_missions: int) -> torch.Tensor:
    """UAV 간 미션 분배 균형에 대한 페널티 계산
    
    Args:
        assigned_missions: UAV별 할당된 미션 목록
        num_missions: 총 미션 수
    
    Returns:
        penalty: 미션 분배 불균형 페널티
    """
    if not assigned_missions or not assigned_missions[0]:
        return torch.tensor(0.0)
    
    # UAV별 미션 수
    mission_counts = torch.tensor([len(missions) for missions in assigned_missions])
    
    if mission_counts.sum() == 0:
        return torch.tensor(0.0)
    
    # 이상적인 미션 분배 (균등 분배)
    ideal_count = num_missions / len(assigned_missions)
    
    # 실제 분배와 이상적 분배의 차이 제곱합
    imbalance = torch.sum((mission_counts - ideal_count) ** 2) / len(assigned_missions)
    
    return imbalance

# 데이터 및 환경 클래스 - 기능 및 효율성 개선
class MissionData:
    """미션 데이터 생성 및 관리"""
    def __init__(self, 
                 num_missions: int = 20, 
                 num_uavs: int = 3, 
                 seed: Optional[int] = None,
                 device: torch.device = torch.device('cpu'),
                 area_size: float = 100.0,
                 fixed_wing_ratio: float = 0.5,
                 fixed_speed_range: Tuple[float, float] = (30.0, 60.0),
                 rotary_speed_range: Tuple[float, float] = (5.0, 15.0),
                 risk_area_range: Tuple[int, int] = (1, 5),
                 risk_radius_range: Tuple[float, float] = (5.0, 15.0),
                 no_entry_range: Tuple[int, int] = (1, 3),
                 no_entry_radius_range: Tuple[float, float] = (3.0, 10.0)):
        """
        Args:
            num_missions: 미션의 총 수 (시작점, 끝점 포함)
            num_uavs: UAV의 수
            seed: 랜덤 시드
            device: 텐서 계산 장치
            area_size: 작전 영역 크기
            fixed_wing_ratio: 고정익 UAV 비율
            fixed_speed_range: 고정익 UAV 속도 범위 (최소, 최대)
            rotary_speed_range: 회전익 UAV 속도 범위 (최소, 최대)
            risk_area_range: 위험 지역 수 범위 (최소, 최대)
            risk_radius_range: 위험 지역 반경 범위 (최소, 최대)
            no_entry_range: 출입 불가 지역 수 범위 (최소, 최대)
            no_entry_radius_range: 출입 불가 지역 반경 범위 (최소, 최대)
        """
        self.num_missions = num_missions
        self.num_uavs = num_uavs
        self.seed = seed
        self.device = device
        self.area_size = area_size
        self.fixed_wing_ratio = fixed_wing_ratio
        self.fixed_speed_range = fixed_speed_range
        self.rotary_speed_range = rotary_speed_range
        self.risk_area_range = risk_area_range
        self.risk_radius_range = risk_radius_range
        self.no_entry_range = no_entry_range
        self.no_entry_radius_range = no_entry_radius_range
        
        # 랜덤 시드 설정
        self._set_seed()
        
        # 미션 및 UAV 데이터 생성
        self.missions, self.uavs_start, self.uavs_end, self.uavs_speeds, self.uav_types = self._generate_mission_data()
        
        # 위험 지역 및 출입 불가 지역 생성
        self.risk_areas = self._generate_risk_areas()
        self.no_entry_zones = self._generate_no_entry_zones()
        
        # 위험 지역 및 출입 불가 지역 텐서 변환
        self.risk_centers, self.risk_radii = self._create_area_tensors(self.risk_areas)
        self.zone_centers, self.zone_radii = self._create_area_tensors(self.no_entry_zones)
    
    def _set_seed(self) -> None:
        """랜덤 시드 설정"""
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
    
    def _generate_mission_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """미션 및 UAV 데이터 생성"""
        # 미션 위치 생성 (미들 미션)
        missions = torch.rand((self.num_missions - 2, 2), device=self.device) * self.area_size
        
        # 시작점 및 종료점 생성
        start_point = torch.rand((1, 2), device=self.device) * self.area_size
        end_point = torch.rand((1, 2), device=self.device) * self.area_size
        
        # 모든 미션 좌표 병합
        missions = torch.cat([start_point, missions, end_point], dim=0)
        
        # UAV 시작 및 종료 위치 설정 (모든 UAV가 동일한 위치에서 시작하고 종료)
        uavs_start = start_point.repeat(self.num_uavs, 1)
        uavs_end = end_point.repeat(self.num_uavs, 1)
        
        # UAV 유형 및 속도 설정
        # 고정익: 유형 0, 회전익: 유형 1
        num_fixed = max(1, int(self.num_uavs * self.fixed_wing_ratio))
        num_rotary = self.num_uavs - num_fixed
        
        # UAV 유형 텐서 생성
        uav_types = torch.cat([
            torch.zeros(num_fixed), 
            torch.ones(num_rotary)
        ]).to(self.device)
        
        # 속도 생성
        fixed_min, fixed_max = self.fixed_speed_range
        rotary_min, rotary_max = self.rotary_speed_range
        
        fixed_speeds = torch.randint(
            int(fixed_min), int(fixed_max) + 1, 
            (num_fixed,), device=self.device
        ).float()
        
        rotary_speeds = torch.randint(
            int(rotary_min), int(rotary_max) + 1, 
            (num_rotary,), device=self.device
        ).float()
        
        # 속도 텐서 병합
        uavs_speeds = torch.cat([fixed_speeds, rotary_speeds])
        
        return missions, uavs_start, uavs_end, uavs_speeds, uav_types
    
    def _generate_risk_areas(self) -> List[Dict[str, Union[torch.Tensor, float]]]:
        """위험 지역 생성"""
        min_areas, max_areas = self.risk_area_range
        min_radius, max_radius = self.risk_radius_range
        
        num_risk_areas = random.randint(min_areas, max_areas)
        
        return [{
            'center': torch.rand(2, device=self.device) * self.area_size,
            'radius': random.uniform(min_radius, max_radius),
            'risk_level': random.uniform(0.5, 1.0)  # 위험도 추가
        } for _ in range(num_risk_areas)]
    
    def _generate_no_entry_zones(self) -> List[Dict[str, Union[torch.Tensor, float]]]:
        """출입 불가 지역 생성"""
        min_zones, max_zones = self.no_entry_range
        min_radius, max_radius = self.no_entry_radius_range
        
        num_no_entry_zones = random.randint(min_zones, max_zones)
        
        return [{
            'center': torch.rand(2, device=self.device) * self.area_size,
            'radius': random.uniform(min_radius, max_radius)
        } for _ in range(num_no_entry_zones)]
    
    def _create_area_tensors(self, areas: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """영역 텐서 생성"""
        if not areas:
            return torch.empty((0, 2), device=self.device), torch.empty(0, device=self.device)
        
        # 중심점과 반경을 분리하여 텐서로 변환
        centers = torch.stack([area['center'] for area in areas])
        radii = torch.tensor([area['radius'] for area in areas], device=self.device)
        
        return centers, radii
    
    def generate_validation_data(self, val_seed: Optional[int] = None) -> 'MissionData':
        """검증 데이터 생성
        
        기존 설정을 유지하면서 다른 시드로 새로운 미션 데이터 생성
        
        Args:
            val_seed: 검증 데이터용 랜덤 시드
        
        Returns:
            validation_data: 새로운 미션 데이터 객체
        """
        if val_seed is None:
            if self.seed is not None:
                val_seed = self.seed + 1000
            else:
                val_seed = random.randint(0, 10000)
        
        return MissionData(
            num_missions=self.num_missions,
            num_uavs=self.num_uavs,
            seed=val_seed,
            device=self.device,
            area_size=self.area_size,
            fixed_wing_ratio=self.fixed_wing_ratio,
            fixed_speed_range=self.fixed_speed_range,
            rotary_speed_range=self.rotary_speed_range,
            risk_area_range=self.risk_area_range,
            risk_radius_range=self.risk_radius_range,
            no_entry_range=self.no_entry_range,
            no_entry_radius_range=self.no_entry_radius_range
        )
    
    def visualize(self, path: Optional[str] = None) -> None:
        """미션 및 위험 지역 시각화
        
        Args:
            path: 이미지 저장 경로, None이면 화면에 표시
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            
            plt.figure(figsize=(10, 10))
            
            # 미션 위치 그리기
            missions_np = self.missions.cpu().numpy()
            plt.scatter(missions_np[1:-1, 0], missions_np[1:-1, 1], c='blue', label='Missions')
            plt.scatter(missions_np[0, 0], missions_np[0, 1], c='green', s=100, label='Start')
            plt.scatter(missions_np[-1, 0], missions_np[-1, 1], c='red', s=100, label='End')
            
            # 위험 지역 그리기
            ax = plt.gca()
            for area in self.risk_areas:
                center = area['center'].cpu().numpy()
                radius = area['radius']
                risk_level = area.get('risk_level', 1.0)
                circle = Circle(center, radius, alpha=0.3, color='red')
                ax.add_patch(circle)
            
            # 출입 불가 지역 그리기
            for zone in self.no_entry_zones:
                center = zone['center'].cpu().numpy()
                radius = zone['radius']
                circle = Circle(center, radius, alpha=0.5, color='black')
                ax.add_patch(circle)
            
            plt.grid(True)
            plt.xlim(0, self.area_size)
            plt.ylim(0, self.area_size)
            plt.legend()
            plt.title(f"Mission Map ({self.num_missions} missions, {self.num_uavs} UAVs)")
            
            if path:
                plt.savefig(path)
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            print("matplotlib이 설치되어 있지 않아 시각화를 수행할 수 없습니다.")


class MissionEnvironment:
    """다중 UAV 미션 환경 - 강화학습 환경으로 기능"""
    def __init__(self, 
                 missions: torch.Tensor, 
                 uavs_start: torch.Tensor, 
                 uavs_end: torch.Tensor,
                 uavs_speeds: torch.Tensor, 
                 uav_types: torch.Tensor, 
                 risk_centers: torch.Tensor,
                 risk_radii: torch.Tensor, 
                 zone_centers: torch.Tensor, 
                 zone_radii: torch.Tensor,
                 device: torch.device, 
                 seed: Optional[int] = None, 
                 curriculum_epoch: Optional[int] = None,
                 total_epochs: Optional[int] = None, 
                 min_missions: int = 5,
                 adaptive_curriculum: bool = False,
                 success_threshold: float = 0.7):
        """
        Args:
            missions: 미션 좌표
            uavs_start: UAV 시작 위치
            uavs_end: UAV 종료 위치
            uavs_speeds: UAV 속도
            uav_types: UAV 유형 (0: 고정익, 1: 회전익)
            risk_centers: 위험 지역 중심점
            risk_radii: 위험 지역 반경
            zone_centers: 출입 불가 지역 중심점
            zone_radii: 출입 불가 지역 반경
            device: 계산 장치
            seed: 랜덤 시드
            curriculum_epoch: 현재 커리큘럼 에포크
            total_epochs: 총 에포크 수
            min_missions: 최소 미션 수
            adaptive_curriculum: 적응형 커리큘럼 사용 여부
            success_threshold: 난이도 상승을 위한 성공률 임계값
        """
        self.device = device
        self.seed = seed
        
        # 미션 및 UAV 데이터 저장
        self.original_missions = missions.clone()
        self.max_missions = missions.size(0)
        self.uavs_start = uavs_start
        self.uavs_end = uavs_end
        self.speeds = uavs_speeds
        self.uav_types = uav_types
        
        # 위험 지역 및 출입 불가 지역 데이터
        self.risk_centers = risk_centers
        self.risk_radii = risk_radii
        self.zone_centers = zone_centers
        self.zone_radii = zone_radii
        
        # 커리큘럼 학습 설정
        self.use_curriculum = curriculum_epoch is not None and total_epochs is not None
        self.curriculum_epoch = curriculum_epoch
        self.total_epochs = total_epochs
        self.min_missions = min_missions
        
        # 적응형 커리큘럼 설정
        self.adaptive_curriculum = adaptive_curriculum
        self.success_threshold = success_threshold
        self.success_rate = 0.0
        self.curriculum_difficulty = min_missions
        
        # 환경 초기화
        self.reset()
        
        # 통계 추적
        self.episode_steps = 0
        self.episode_history = []
    
    def adjust_curriculum(self) -> int:
        """커리큘럼에 따른 미션 수 조정"""
        if not self.use_curriculum:
            return self.max_missions
        
        if self.adaptive_curriculum:
            # 적응형 커리큘럼: 성공률에 따라 난이도 조정
            return self.curriculum_difficulty
        else:
            # 고정 커리큘럼: 에포크에 따라 선형적으로 난이도 증가
            # 80% 지점까지 난이도를 선형적으로 증가시킨 후 최대 난이도 유지
            progress = min(1.0, self.curriculum_epoch / (self.total_epochs * 0.8))
            num_missions = int(self.min_missions + (self.max_missions - self.min_missions) * progress)
            return min(max(self.min_missions, num_missions), self.max_missions)
    
    def update_curriculum_difficulty(self, success: bool) -> None:
        """적응형 커리큘럼 난이도 업데이트
        
        Args:
            success: 에피소드 성공 여부
        """
        if not self.adaptive_curriculum:
            return
        
        # 이동 평균으로 성공률 업데이트 (가중치 0.9)
        self.success_rate = 0.9 * self.success_rate + 0.1 * float(success)
        
        # 성공률이 임계값을 넘으면 난이도 증가
        if self.success_rate > self.success_threshold:
            # 현재 난이도와 최대 난이도 사이의 중간점으로 증가
            new_difficulty = min(self.curriculum_difficulty + 1, self.max_missions)
            if new_difficulty != self.curriculum_difficulty:
                self.curriculum_difficulty = new_difficulty
                self.success_rate = max(0.0, self.success_rate - 0.2)  # 성공률 감소
        
        # 성공률이 매우 낮으면 난이도 감소
        elif self.success_rate < 0.3 and self.curriculum_difficulty > self.min_missions:
            self.curriculum_difficulty = max(self.min_missions, self.curriculum_difficulty - 1)
            self.success_rate = min(1.0, self.success_rate + 0.1)  # 성공률 증가
    
    def reset(self, clone: bool = True) -> Dict[str, torch.Tensor]:
        """환경 초기화
        
        Args:
            clone: 상태 텐서 복제 여부
        
        Returns:
            state: 초기화된 환경 상태
        """
        # 커리큘럼에 따른 미션 수 조정
        num_missions = self.adjust_curriculum()
        
        if num_missions < self.max_missions:
            # 필수 미션 (시작, 종료)
            selected_indices = [0, self.max_missions - 1]
            
            # 중간 미션 무작위 선택
            middle_count = num_missions - 2
            if middle_count > 0:
                # 1부터 max_missions-2까지의 인덱스 중에서 무작위 선택
                middle_indices = random.sample(
                    range(1, self.max_missions - 1), 
                    min(middle_count, self.max_missions - 2)
                )
                # 시작-중간-종료 순서로 정렬
                selected_indices = [0] + sorted(middle_indices) + [self.max_missions - 1]
            
            # 선택된 미션만 사용
            self.missions = self.original_missions[selected_indices].clone()
        else:
            # 모든 미션 사용
            self.missions = self.original_missions.clone()
        
        self.num_missions = self.missions.size(0)
        self.num_uavs = self.uavs_start.size(0)
        
        # 환경 상태 초기화
        self.current_positions = self.uavs_start.clone()
        self.visited = torch.zeros(self.num_missions, dtype=torch.bool, device=self.device)
        self.visited[0] = True  # 시작점은 이미 방문
        self.reserved = torch.zeros_like(self.visited)
        self.paths = [[] for _ in range(self.num_uavs)]
        self.cumulative_travel_times = torch.zeros(self.num_uavs, device=self.device)
        self.ready_for_next_action = torch.ones(self.num_uavs, dtype=torch.bool, device=self.device)
        self.targets = torch.full((self.num_uavs,), -1, dtype=torch.long, device=self.device)
        self.remaining_distances = torch.full((self.num_uavs,), float('inf'), device=self.device)
        self.assigned_missions = [[] for _ in range(self.num_uavs)]
        
        # 통계 초기화
        self.episode_steps = 0
        
        return self.get_state(clone)
    
    def get_state(self, clone: bool = True) -> Dict[str, torch.Tensor]:
        """현재 상태 반환
        
        Args:
            clone: 상태 텐서 복제 여부
        
        Returns:
            state: 현재 환경 상태
        """
        if clone:
            return {
                'positions': self.current_positions.clone(),
                'visited': self.visited.clone(),
                'reserved': self.reserved.clone(),
                'ready_for_next_action': self.ready_for_next_action.clone(),
                'cumulative_times': self.cumulative_travel_times.clone(),
                'targets': self.targets.clone(),
                'missions': self.missions.clone()
            }
        else:
            return {
                'positions': self.current_positions,
                'visited': self.visited,
                'reserved': self.reserved,
                'ready_for_next_action': self.ready_for_next_action,
                'cumulative_times': self.cumulative_travel_times,
                'targets': self.targets,
                'missions': self.missions
            }
    
    def create_action_mask(self) -> torch.Tensor:
        """액션 마스크 생성 - 유효하지 않은 액션 마스킹
        
        Returns:
            mask: 액션 마스크 (True: 유효하지 않음)
        """
        # 기본 마스크: 이미 방문했거나 예약된 미션은 방문 불가
        mask_base = (self.visited | self.reserved).unsqueeze(0).repeat(self.num_uavs, 1)
        
        # 특수 케이스 처리:
        # 1. 중간 미션을 하나도 방문하지 않았다면 종료 지점 방문 불가
        if not self.visited[1:-1].any():
            mask_base[:, -1] = True
        
        # 2. 모든 중간 미션을 방문했다면 종료 지점 방문 가능
        if self.visited[1:-1].all():
            mask_base[:, -1] = False
        
        # 3. 행동할 준비가 되지 않은 UAV는 모든 미션 방문 불가
        mask_base[~self.ready_for_next_action] = True
        
        # 4. 출입 불가 지역과 교차하는 경로는 방문 불가
        if self.zone_centers.shape[0] > 0:
            ready_uavs = torch.where(self.ready_for_next_action)[0]
            if len(ready_uavs) > 0:
                ready_positions = self.current_positions[ready_uavs]
                # 출입 불가 지역과의 교차 계산
                intersections = compute_segment_circle_intersections(
                    ready_positions, self.missions, self.zone_centers, self.zone_radii
                )
                # 교차하는 경로 마스킹
                for i, u_idx in enumerate(ready_uavs):
                    mask_base[u_idx] |= intersections[i]
        
        # 항상 최소한 하나의 유효한 액션 제공
        # (모든 액션이 유효하지 않은 경우, 시작점을 유효하게 설정)
        for u in range(self.num_uavs):
            if mask_base[u].all():
                mask_base[u, 0] = False
        
        return mask_base
    
    def calculate_cost_matrix(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """시간 기반 비용 행렬 계산
        
        Returns:
            timetogo_matrix: UAV별 미션까지의 이동 시간 행렬
            dist_matrix: UAV별 미션까지의 거리 행렬
        """
        # 거리 행렬 계산
        dist_matrix = calculate_distance_matrix(self.current_positions, self.missions)
        
        # 속도로 나누어 시간 계산
        speeds_expanded = self.speeds.unsqueeze(1)
        timetogo_matrix = dist_matrix / (speeds_expanded + 1e-8)
        
        # 위험 지역에 대한 페널티 계산
        if self.risk_centers.shape[0] > 0:
            risk_penalties = calculate_risk_penalty(
                self.current_positions, self.missions, 
                self.risk_centers, self.risk_radii
            )
            # 위험 페널티 적용
            timetogo_matrix += risk_penalties
        
        return timetogo_matrix, dist_matrix
    
    def step(self, actions: List[int]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, bool, Dict[str, Any]]:
        """액션 실행 및 다음 상태 반환 - gym 환경과 유사한 인터페이스
        
        Args:
            actions: UAV별 선택된 미션 인덱스 목록
        
        Returns:
            next_state: 다음 환경 상태
            rewards: UAV별 보상
            done: 에피소드 종료 여부
            info: 추가 정보
        """
        self.episode_steps += 1
        actions = torch.tensor(actions, device=self.device)
        travel_times = torch.zeros(self.num_uavs, device=self.device)
        
        # 액션 적용: 목표 미션 설정
        for u, action in enumerate(actions):
            if not self.ready_for_next_action[u] or action == -1:
                continue
            
            # 아직 방문하지 않았고 예약되지 않은 미션인 경우
            if not self.visited[action] and not self.reserved[action]:
                self.reserved[action] = True
                self.ready_for_next_action[u] = False
                self.targets[u] = action
                # 현재 위치에서 목표 미션까지의 거리 계산
                self.remaining_distances[u] = calculate_distance(
                    self.current_positions[u], 
                    self.missions[action]
                )
        
        # 이동 실행: 목표로 향하는 UAV 이동
        for u, target in enumerate(self.targets):
            if target != -1 and not self.ready_for_next_action[u]:
                # 이동 시간 계산
                travel_time = self.remaining_distances[u] / (self.speeds[u] + 1e-8)
                
                # 누적 이동 시간 업데이트
                self.cumulative_travel_times[u] += travel_time
                
                # 위치 업데이트
                self.current_positions[u] = self.missions[target]
                
                # 목표 달성 처리
                if target != self.num_missions - 1:  # 종료 지점이 아니면 방문 처리
                    self.visited[target] = True
                
                # 미션 할당 기록
                self.assigned_missions[u].append(target.item())
                
                # UAV 상태 업데이트
                self.ready_for_next_action[u] = True
                self.reserved[target] = False
                self.targets[u] = -1
                
                # 이동 시간 기록
                travel_times[u] = travel_time
        
        # 모든 중간 미션이 완료된 경우, 종료 지점으로 향하도록 유도
        all_mid_missions_done = self.visited[1:-1].all().item()
        if all_mid_missions_done:
            for u in range(self.num_uavs):
                # 행동할 준비가 되었고, 아직 종료 지점으로 향하지 않았으며, 종료 지점에 도착하지 않은 UAV
                if (self.ready_for_next_action[u] and 
                    self.targets[u] != self.num_missions - 1 and 
                    not self.visited[-1]):
                    # 종료 지점으로 설정
                    self.targets[u] = self.num_missions - 1
                    self.ready_for_next_action[u] = False
                    self.remaining_distances[u] = calculate_distance(
                        self.current_positions[u], 
                        self.missions[-1]
                    )
        
        # 에피소드 종료 조건: 모든 중간 미션 완료 및 하나 이상의 UAV가 종료 지점에 도착
        done = all_mid_missions_done and any(self.num_missions - 1 in path for path in self.assigned_missions)
        
        # 추가 정보
        info = {
            'travel_times': travel_times,
            'assigned_missions': self.assigned_missions,
            'all_mid_missions_done': all_mid_missions_done,
            'steps': self.episode_steps
        }
        
        if done:
            # 성공 여부 기록
            self.update_curriculum_difficulty(True)
            info['success'] = True
            
            # 에피소드 통계 기록
            self.episode_history.append({
                'steps': self.episode_steps,
                'total_time': self.cumulative_travel_times.sum().item(),
                'max_time': self.cumulative_travel_times.max().item(),
                'success': True,
                'missions_completed': sum(self.visited.cpu().numpy())
            })
        elif self.episode_steps >= 100:  # 최대 스텝 제한
            # 실패로 판단
            self.update_curriculum_difficulty(False)
            done = True
            info['success'] = False
            
            # 에피소드 통계 기록
            self.episode_history.append({
                'steps': self.episode_steps,
                'total_time': self.cumulative_travel_times.sum().item(),
                'max_time': self.cumulative_travel_times.max().item(),
                'success': False,
                'missions_completed': sum(self.visited.cpu().numpy())
            })
        
        return self.get_state(), travel_times, done, info
    
    def seed(self, seed: Optional[int] = None) -> int:
        """환경의 랜덤 시드 설정"""
        if seed is not None:
            self.seed = seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        return self.seed
    
    def get_all_mission_assignments(self) -> Dict[int, List[int]]:
        """모든 UAV의 미션 할당 현황 반환"""
        return {i: missions for i, missions in enumerate(self.assigned_missions)}
    
    def visualize_paths(self, path: Optional[str] = None) -> None:
        """UAV 경로 시각화
        
        Args:
            path: 이미지 저장 경로, None이면 화면에 표시
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            
            plt.figure(figsize=(12, 10))
            
            # 미션 위치 그리기
            missions_np = self.missions.cpu().numpy()
            plt.scatter(missions_np[1:-1, 0], missions_np[1:-1, 1], c='blue', label='Missions')
            plt.scatter(missions_np[0, 0], missions_np[0, 1], c='green', s=100, label='Start')
            plt.scatter(missions_np[-1, 0], missions_np[-1, 1], c='red', s=100, label='End')
            
            # 위험 지역 그리기
            ax = plt.gca()
            if self.risk_centers.shape[0] > 0:
                for i in range(self.risk_centers.shape[0]):
                    center = self.risk_centers[i].cpu().numpy()
                    radius = self.risk_radii[i].item()
                    circle = Circle(center, radius, alpha=0.3, color='red')
                    ax.add_patch(circle)
            
            # 출입 불가 지역 그리기
            if self.zone_centers.shape[0] > 0:
                for i in range(self.zone_centers.shape[0]):
                    center = self.zone_centers[i].cpu().numpy()
                    radius = self.zone_radii[i].item()
                    circle = Circle(center, radius, alpha=0.5, color='black')
                    ax.add_patch(circle)
            
            # UAV 경로 그리기
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            for u, mission_indices in enumerate(self.assigned_missions):
                if not mission_indices:
                    continue
                
                # 경로 좌표 추출
                path_coords = [self.missions[idx].cpu().numpy() for idx in mission_indices]
                if mission_indices[0] != 0:  # 시작점이 포함되지 않은 경우 추가
                    path_coords.insert(0, self.missions[0].cpu().numpy())
                
                # 마지막 위치가 현재 위치와 다른 경우 (미완료 경로)
                if len(path_coords) > 0 and self.current_positions[u].cpu().numpy().tolist() != path_coords[-1].tolist():
                    path_coords.append(self.current_positions[u].cpu().numpy())
                
                # 경로 그리기
                path_x, path_y = zip(*path_coords)
                plt.plot(path_x, path_y, '-o', color=colors[u % len(colors)], 
                         label=f'UAV {u} ({"Fixed" if self.uav_types[u]==0 else "Rotary"})')
            
            plt.grid(True)
            plt.xlim(0, 100)
            plt.ylim(0, 100)
            plt.legend()
            plt.title(f"UAV Paths (Curriculum lvl: {self.curriculum_difficulty}/{self.max_missions})")
            
            if path:
                plt.savefig(path)
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            print("matplotlib이 설치되어 있지 않아 시각화를 수행할 수 없습니다.")

# 신경망 정의 - 개선된 아키텍처 및 모듈화
class PositionalEncoding(nn.Module):
    """Transformer 위치 인코딩"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 위치 인코딩 생성
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # 등록된 버퍼는 모델의 상태 딕셔너리에 저장되지만 파라미터로 간주되지 않음
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model] 또는 [batch_size, seq_len, d_model]
        """
        if x.dim() == 3 and x.size(0) != self.pe.size(0):  # [batch_size, seq_len, d_model] 형태
            x = x.transpose(0, 1)
            transposed = True
        else:
            transposed = False
        
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        
        if transposed:
            x = x.transpose(0, 1)
        
        return x

class GraphEncoder(nn.Module):
    """그래프 인코더 - 메시지 패싱을 통한 노드 특성 인코딩"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3, 
                heads: int = 8, dropout: float = 0.1, residual: bool = True):
        super().__init__()
        
        # 입력 투영
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT 레이어
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # 다중 헤드 어텐션을 통한 메시지 패싱
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            )
            # 레이어 정규화
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # 설정 저장
        self.residual = residual
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                node_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 노드 특성 [N, F_in]
            edge_index: 엣지 인덱스 [2, E]
            node_types: 노드 타입 [N]
        
        Returns:
            encoded_features: 인코딩된 노드 특성 [N, hidden_dim]
        """
        # 입력 투영
        x = self.input_proj(x)
        
        # 여러 레이어를 통한 메시지 패싱
        for layer, norm in zip(self.gat_layers, self.layer_norms):
            if self.residual:
                # 잔차 연결
                x_res = x
            
            # 그래프 어텐션 레이어
            x = layer(x, edge_index)
            x = F.relu(x)
            
            if self.residual:
                # 잔차 연결 및 정규화
                x = norm(x + x_res)
            else:
                # 정규화만 적용
                x = norm(x)
            
            # 훈련 중 드롭아웃 적용
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

class MissionEmbedding(nn.Module):
    """미션 및 UAV 임베딩 - 특성 추출 및 임베딩"""
    def __init__(self, input_dim: int, hidden_dim: int, 
                 max_missions: int, max_uavs: int = 10, dropout: float = 0.1):
        super().__init__()
        
        # 미션 좌표 임베딩
        self.mission_embedding = nn.Linear(input_dim, hidden_dim)
        
        # 미션 ID 임베딩 (위치 구분을 위한 학습 가능한 임베딩)
        self.mission_id_embedding = nn.Embedding(max_missions, hidden_dim)
        
        # UAV 특성 임베딩 (위치 + 속도 + UAV 타입)
        self.uav_embedding = nn.Linear(input_dim + 2, hidden_dim)  # +2: 속도, 유형
        
        # UAV ID 임베딩
        self.uav_id_embedding = nn.Embedding(max_uavs, hidden_dim)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                missions: torch.Tensor, 
                uav_positions: torch.Tensor, 
                uav_speeds: torch.Tensor,
                uav_types: Optional[torch.Tensor] = None,
                mission_ids: Optional[torch.Tensor] = None,
                uav_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            missions: 미션 좌표 [num_missions, 2]
            uav_positions: UAV 위치 [num_uavs, 2]
            uav_speeds: UAV 속도 [num_uavs]
            uav_types: UAV 유형 [num_uavs] (옵션)
            mission_ids: 미션 ID [num_missions] (옵션)
            uav_ids: UAV ID [num_uavs] (옵션)
        
        Returns:
            mission_emb: 미션 임베딩 [num_missions, hidden_dim]
            uav_emb: UAV 임베딩 [num_uavs, hidden_dim]
        """
        # 기본 ID 생성 (제공되지 않은 경우)
        if mission_ids is None:
            mission_ids = torch.arange(missions.size(0), device=missions.device)
        
        if uav_ids is None:
            uav_ids = torch.arange(uav_positions.size(0), device=uav_positions.device)
        
        # 미션 임베딩 계산
        mission_coord_emb = self.mission_embedding(missions)
        mission_id_emb = self.mission_id_embedding(mission_ids)
        mission_emb = mission_coord_emb + mission_id_emb
        
        # UAV 특성 결합
        uav_speeds_expanded = uav_speeds.unsqueeze(1)
        
        if uav_types is not None:
            uav_types_expanded = uav_types.unsqueeze(1)
            uav_features = torch.cat([uav_positions, uav_speeds_expanded, uav_types_expanded], dim=1)
        else:
            # 기본값으로 0 (고정익)을 사용
            uav_types_expanded = torch.zeros_like(uav_speeds_expanded)
            uav_features = torch.cat([uav_positions, uav_speeds_expanded, uav_types_expanded], dim=1)
        
        # UAV 임베딩 계산
        uav_feature_emb = self.uav_embedding(uav_features)
        uav_id_emb = self.uav_id_embedding(uav_ids)
        uav_emb = uav_feature_emb + uav_id_emb
        
        # 드롭아웃 적용
        return self.dropout(mission_emb), self.dropout(uav_emb)

class CustomGRU(nn.Module):
    """커스텀 GRU 셀 - 경로 시퀀스 처리를 위한 RNN"""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 입력 시퀀스 [batch_size, seq_len, input_dim]
            hidden: 초기 은닉 상태 [1, batch_size, hidden_dim]
        
        Returns:
            output: 모든 시간 단계의 출력 [batch_size, seq_len, hidden_dim]
            hidden: 최종 은닉 상태 [1, batch_size, hidden_dim]
        """
        output, hidden = self.gru(x, hidden)
        return self.dropout(output), hidden

class TransformerDecoder(nn.Module):
    """Transformer 디코더 - 미션 시퀀스 처리"""
    def __init__(self, d_model: int, nhead: int, num_layers: int = 3, 
                dim_feedforward: int = 2048, dropout: float = 0.1, 
                activation: str = "relu"):
        super().__init__()
        
        # 트랜스포머 디코더 레이어
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # 최종 정규화
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt: 디코더 입력 [batch_size, tgt_len, d_model]
            memory: 인코더 출력 [batch_size, src_len, d_model]
            tgt_mask: 타겟 시퀀스 마스크 [tgt_len, tgt_len]
            tgt_key_padding_mask: 패딩 마스크 [batch_size, tgt_len]
        
        Returns:
            output: 디코더 출력 [batch_size, tgt_len, d_model]
        """
        output = tgt
        
        # 여러 디코더 레이어 적용
        for layer in self.layers:
            output = layer(
                output, memory, 
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
        
        # 최종 정규화
        return self.norm(output)

class AttentionPooling(nn.Module):
    """어텐션 풀링 - 가변 길이 시퀀스에서 중요 정보 추출"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 입력 시퀀스 [batch_size, seq_len, hidden_dim]
            mask: 시퀀스 마스크 [batch_size, seq_len]
        
        Returns:
            weighted_sum: 어텐션 가중 합 [batch_size, hidden_dim]
        """
        # 어텐션 점수 계산
        attn_scores = self.attention(x).squeeze(-1)  # [batch_size, seq_len]
        
        # 마스크 적용
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -1e9)
        
        # 소프트맥스로 정규화
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(1)  # [batch_size, 1, seq_len]
        
        # 가중 합 계산
        weighted_sum = torch.bmm(attn_weights, x).squeeze(1)  # [batch_size, hidden_dim]
        
        return weighted_sum

class TransformerAllocationNetwork(nn.Module):
    """Transformer 기반 액터 네트워크 - 미션 할당 예측"""
    def __init__(self, 
                 input_dim: int = 2, 
                 hidden_dim: int = 128, 
                 num_gat_layers: int = 3, 
                 gat_heads: int = 8,
                 num_transformer_layers: int = 3, 
                 transformer_heads: int = 8, 
                 max_missions: int = 100, 
                 max_uavs: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        
        # 임베딩 레이어
        self.embedding = MissionEmbedding(
            input_dim, hidden_dim, max_missions, max_uavs, dropout
        )
        
        # 그래프 인코더
        self.graph_encoder = GraphEncoder(
            hidden_dim, hidden_dim, num_gat_layers, gat_heads, dropout
        )
        
        # 위치 인코딩
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # 경로 인코더 (GRU)
        self.path_encoder = CustomGRU(hidden_dim, hidden_dim, dropout)
        
        # 트랜스포머 디코더
        self.transformer_decoder = TransformerDecoder(
            hidden_dim, transformer_heads, num_transformer_layers,
            hidden_dim * 4, dropout
        )
        
        # 어텐션 풀링
        self.attention_pooling = AttentionPooling(hidden_dim)
        
        # 쿼리 생성 레이어
        self.query_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 설정 저장
        self.hidden_dim = hidden_dim

    def forward(self, 
                missions: torch.Tensor, 
                edge_index: torch.Tensor, 
                uav_positions: torch.Tensor,
                uav_speeds: torch.Tensor, 
                uav_types: torch.Tensor,
                assigned_missions: List[List[int]], 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            missions: 미션 좌표 [num_missions, 2]
            edge_index: 그래프 엣지 인덱스 [2, num_edges]
            uav_positions: UAV 위치 [num_uavs, 2]
            uav_speeds: UAV 속도 [num_uavs]
            uav_types: UAV 타입 [num_uavs]
            assigned_missions: UAV별 할당된 미션 목록
            mask: 액션 마스크 [num_uavs, num_missions]
        
        Returns:
            action_logits: 각 UAV의 다음 미션 선택 로짓 [num_uavs, num_missions]
        """
        num_missions = missions.size(0)
        num_uavs = uav_positions.size(0)
        device = missions.device
        
        # 미션 및 UAV 임베딩
        mission_emb, uav_emb = self.embedding(
            missions, uav_positions, uav_speeds, uav_types
        )
        
        # 미션 그래프 인코딩
        node_types = torch.zeros(num_missions, dtype=torch.long, device=device)
        mission_encoding = self.graph_encoder(mission_emb, edge_index, node_types)
        
        # 경로 시퀀스 데이터 준비
        max_seq_len = max([len(seq) for seq in assigned_missions], default=0) + 1
        seq_tensor = torch.full(
            (num_uavs, max_seq_len), -1, dtype=torch.long, device=device
        )
        
        # 할당된 미션 시퀀스 채우기
        for i, seq in enumerate(assigned_missions):
            if seq:
                seq_tensor[i, :len(seq)] = torch.tensor(seq, device=device)
        
        # 패딩 마스크 생성
        padding_mask = (seq_tensor == -1)
        
        # 시퀀스 표현 생성
        path_features = torch.zeros(
            num_uavs, max_seq_len, self.hidden_dim, device=device
        )
        
        # UAV의 경로 특성 생성
        for i in range(num_uavs):
            # 첫 번째 토큰은 UAV 특성
            path_features[i, 0] = uav_emb[i]
            
            # 나머지 토큰은 방문한 미션 특성
            for j in range(1, max_seq_len):
                if not padding_mask[i, j]:
                    mission_idx = seq_tensor[i, j-1].item()
                    if 0 <= mission_idx < num_missions:
                        path_features[i, j] = mission_encoding[mission_idx]
        
        # 위치 인코딩 적용
        path_features = self.pos_encoder(path_features)
        
        # 트랜스포머 마스크 생성 (미래 토큰 마스킹)
        tgt_mask = get_subsequent_mask(max_seq_len, device)
        
        # 메모리 확장 (각 UAV에 대해 동일한 미션 인코딩 제공)
        memory = mission_encoding.unsqueeze(0).expand(num_uavs, -1, -1)
        
        # 트랜스포머 디코더 적용
        decoder_output = self.transformer_decoder(
            path_features, memory, tgt_mask=tgt_mask, 
            tgt_key_padding_mask=padding_mask
        )
        
        # 시퀀스 길이 계산 (패딩 제외)
        seq_lens = max_seq_len - padding_mask.sum(dim=1)
        
        # UAV별 쿼리 생성
        queries = torch.zeros(num_uavs, self.hidden_dim, device=device)
        last_state_features = torch.zeros(num_uavs, self.hidden_dim, device=device)
        
        for i in range(num_uavs):
            # 마지막 유효 토큰 선택
            idx = max(0, seq_lens[i] - 1)
            # 디코더 출력 및 UAV 특성 결합
            last_state_features[i] = decoder_output[i, idx]
        
        # UAV 특성과 마지막 상태 결합하여 쿼리 생성
        combined_features = torch.cat([uav_emb, last_state_features], dim=1)
        queries = self.query_generator(combined_features)
        
        # 쿼리와 미션 인코딩 간 어텐션 점수 계산
        logits = torch.matmul(queries, mission_encoding.T)
        
        # 마스크 적용
        if mask is not None:
            logits = logits.masked_fill(mask, -1e9)
        
        return logits

class SurrogateNetwork(nn.Module):
    """크리틱 네트워크 - 상태 가치 추정"""
    def __init__(self, 
                 input_dim: int = 2, 
                 hidden_dim: int = 128, 
                 num_layers: int = 3, 
                 heads: int = 8, 
                 dropout: float = 0.1):
        super().__init__()
        
        # 입력 투영
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 그래프 어텐션 레이어
        self.embedding_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.embedding_layers.append(
                GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # 글로벌 풀링 후 가치 예측을 위한 MLP
        self.cost_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 엣지 특성 인코더 (옵션)
        self.use_edge_features = False
        
    def enable_edge_features(self, edge_dim: int = 1):
        """엣지 특성 사용 설정 (위험도 등을 엣지에 반영)"""
        self.use_edge_features = True
        # TODO: edge_attr 활용 GAT 구현

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                node_types: torch.Tensor, batch=None, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 노드 특성 [num_nodes, input_dim]
            edge_index: 엣지 인덱스 [2, num_edges]
            node_types: 노드 타입 [num_nodes]
            batch: 배치 인덱스 [num_nodes]
            edge_attr: 엣지 특성 [num_edges, edge_dim]
        
        Returns:
            state_value: 상태 가치 추정치 [1] 또는 [batch_size, 1]
        """
        # 입력 투영
        x = self.input_proj(x)
        
        # 그래프 임베딩 레이어 적용
        for layer, norm in zip(self.embedding_layers, self.layer_norms):
            # 잔차 연결
            x_res = x
            
            # 그래프 어텐션 계산
            x = layer(x, edge_index)
            x = F.relu(x)
            
            # 잔차 연결 및 정규화
            x = norm(x + x_res)
        
        # 그래프 풀링 (여러 노드를 하나의 그래프 표현으로)
        if batch is not None:
            try:
                from torch_geometric.nn import global_mean_pool
                x = global_mean_pool(x, batch)
            except ImportError:
                # 배치가 제공되었지만 PyG가 설치되지 않은 경우
                # 간단한 평균 풀링으로 대체
                unique_batches = torch.unique(batch)
                pooled_x = torch.zeros(
                    len(unique_batches), x.size(1), 
                    device=x.device
                )
                
                for i, b in enumerate(unique_batches):
                    mask = (batch == b)
                    pooled_x[i] = x[mask].mean(dim=0)
                
                x = pooled_x
        
        # 가치 예측
        return self.cost_predictor(x)

class TransformerActorCriticNetwork(nn.Module):
    """액터-크리틱 네트워크 - 정책과 가치 함수 결합"""
    def __init__(self, 
                 max_missions: int, 
                 max_uavs: int, 
                 hidden_dim: int = 128, 
                 num_gat_layers: int = 3,
                 gat_heads: int = 8, 
                 dropout: float = 0.1, 
                 transformer_heads: int = 8, 
                 transformer_layers: int = 3):
        super().__init__()
        
        # 액터 네트워크
        self.actor = TransformerAllocationNetwork(
            2, hidden_dim, num_gat_layers, gat_heads, 
            transformer_layers, transformer_heads, 
            max_missions, max_uavs, dropout
        )
        
        # 크리틱 네트워크
        self.critic = SurrogateNetwork(
            2, hidden_dim, num_gat_layers, gat_heads, dropout
        )
        
        # 설정 저장
        self.max_missions = max_missions
        self.max_uavs = max_uavs
        self.hidden_dim = hidden_dim

    def forward(self, 
                missions: torch.Tensor, 
                edge_index: torch.Tensor, 
                batch, 
                uav_positions: torch.Tensor,
                uav_speeds: torch.Tensor, 
                uav_types: torch.Tensor,
                action_mask: torch.Tensor, 
                assigned_missions: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            missions: 미션 좌표 [num_missions, 2]
            edge_index: 그래프 엣지 인덱스 [2, num_edges]
            batch: 배치 인덱스 [num_missions]
            uav_positions: UAV 위치 [num_uavs, 2]
            uav_speeds: UAV 속도 [num_uavs]
            uav_types: UAV 타입 [num_uavs]
            action_mask: 액션 마스크 [num_uavs, num_missions]
            assigned_missions: UAV별 할당된 미션 목록
        
        Returns:
            action_logits: 각 UAV의 다음 미션 선택 로짓 [num_uavs, num_missions]
            state_value: 상태 가치 추정치 [1]
        """
        # 액터 네트워크로 액션 로짓 계산
        action_logits = self.actor(
            missions, edge_index, 
            uav_positions, uav_speeds, uav_types,
            assigned_missions, action_mask
        )
        
        # 크리틱 네트워크로 상태 가치 계산
        node_types = torch.zeros(missions.size(0), dtype=torch.long, device=missions.device)
        state_value = self.critic(missions, edge_index, node_types, batch)
        
        return action_logits, state_value

    def get_actor(self) -> TransformerAllocationNetwork:
        """액터 네트워크 반환"""
        return self.actor
    
    def get_critic(self) -> SurrogateNetwork:
        """크리틱 네트워크 반환"""
        return self.critic
    
# 학습 보조 함수 - 개선된 API와 효율성
def choose_action(
        action_logits: torch.Tensor, 
        temperature: float, 
        action_mask: torch.Tensor,
        deterministic: bool = False
    ) -> List[int]:
    """액션 샘플링 - 확률적 또는 결정적 선택
    
    Args:
        action_logits: 액션 로짓 [num_uavs, num_missions]
        temperature: 샘플링 온도 (낮을수록 더 결정적)
        action_mask: 액션 마스크 [num_uavs, num_missions] (True: 유효하지 않음)
        deterministic: 결정적 정책 사용 여부
    
    Returns:
        actions: 선택된 액션 (미션 인덱스) 목록
    """
    # 온도로 나누어 소프트맥스 계산 (온도가 낮을수록 더 뾰족한 분포)
    logits = action_logits / max(temperature, 1e-8)
    probs = F.softmax(logits, dim=-1)
    
    actions = []
    for i in range(action_logits.size(0)):
        # 모든 액션이 마스킹된 경우
        if action_mask[i].all():
            actions.append(-1)
        else:
            if deterministic:
                # 결정적 정책: 최대 확률 액션 선택
                valid_probs = probs[i].clone()
                valid_probs[action_mask[i]] = 0.0
                action = valid_probs.argmax().item()
            else:
                # 확률적 정책: 확률에 따라 샘플링
                action = torch.multinomial(probs[i], 1).item()
                
                # 마스킹된 액션이 선택된 경우 (드물게 발생 가능)
                if action_mask[i, action]:
                    # 유효한 액션 중에서 재샘플링
                    valid_indices = torch.where(~action_mask[i])[0]
                    if len(valid_indices) > 0:
                        idx = torch.randint(0, len(valid_indices), (1,)).item()
                        action = valid_indices[idx].item()
                    else:
                        action = -1  # 유효한 액션이 없는 경우
            
            actions.append(action)
    
    return actions

# compute_episode_reward 함수 오류 수정 (config가 None인 경우)
def compute_episode_reward(env: 'MissionEnvironment', config: Optional['TrainingConfig'] = None) -> torch.Tensor:
    """MISOCP 형태의 시간 기반 보상 계산 - 정규화 및 가중치 개선
    
    Args:
        env: 미션 환경
        config: 학습 설정 (None인 경우 기본값 사용)
    
    Returns:
        reward: 에피소드 보상
    """
    # 이동 시간 정보
    travel_times = env.cumulative_travel_times
    
    # 기본 통계
    total_time = travel_times.sum()
    time_std = travel_times.std() if travel_times.size(0) > 1 else torch.tensor(0.0, device=env.device)
    max_time = travel_times.max()
    
    # 기본 가중치 설정 (config가 None인 경우)
    alpha = 0.5
    beta = 0.3
    gamma = 0.1
    delta = 0.2
    risk_penalty = 10.0
    
    # config에서 가중치 가져오기 (있는 경우)
    if config is not None:
        alpha = config.alpha
        beta = config.beta
        gamma = config.gamma
        delta = config.delta
        risk_penalty = config.risk_penalty
    
    # 정규화된 구성 요소 계산
    norm_total, norm_std, norm_max = normalize_reward_components(
        total_time, time_std, max_time, env.num_uavs
    )
    
    # 미션 분배 균형 페널티
    balance_penalty = compute_mission_balance_penalty(
        env.assigned_missions, env.num_missions
    )
    
    # 완료되지 않은 미션 페널티
    remaining_penalty = torch.sum(~env.visited).float() * risk_penalty
    
    # 최종 보상 계산
    reward = -(
        alpha * norm_total + 
        beta * norm_std + 
        gamma * norm_max + 
        delta * balance_penalty +
        remaining_penalty
    )
    
    return reward

def log_metrics(
        env: 'MissionEnvironment', 
        global_step: int, 
        episode_reward: float,
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None,
        entropy: Optional[float] = None,
        temperature: Optional[float] = None,
        learning_rate: Optional[float] = None,
        success: Optional[bool] = None
    ) -> Dict[str, Any]:
    """학습 지표 로깅 및 계산
    
    Args:
        env: 미션 환경
        global_step: 전역 스텝 수
        episode_reward: 에피소드 보상
        policy_loss: 정책 손실
        value_loss: 가치 손실
        entropy: 정책 엔트로피
        temperature: 샘플링 온도
        learning_rate: 학습률
        success: 성공 여부
    
    Returns:
        metrics: 로깅할 지표 딕셔너리
    """
    metrics = {
        "global_step": global_step,
        "reward": episode_reward,
    }
    
    # 선택적 지표 추가
    if policy_loss is not None:
        metrics["policy_loss"] = policy_loss
    
    if value_loss is not None:
        metrics["value_loss"] = value_loss
    
    if entropy is not None:
        metrics["entropy"] = entropy
    
    if temperature is not None:
        metrics["temperature"] = temperature
    
    if learning_rate is not None:
        metrics["learning_rate"] = learning_rate
    
    # 환경 통계
    if env.cumulative_travel_times.numel() > 0:
        metrics["total_travel_time"] = env.cumulative_travel_times.sum().item()
        metrics["max_travel_time"] = env.cumulative_travel_times.max().item()
        
        if env.cumulative_travel_times.numel() > 1:
            metrics["time_std"] = env.cumulative_travel_times.std().item()
    
    metrics["visited_missions"] = env.visited.sum().item()
    metrics["total_missions"] = env.num_missions
    
    # 미션 성공률
    if success is not None:
        metrics["success"] = float(success)
    
    return metrics

def calculate_returns(
        rewards: List[torch.Tensor], 
        gamma: float = 0.99
    ) -> torch.Tensor:
    """할인된 수익 계산
    
    Args:
        rewards: 보상 리스트
        gamma: 할인율
    
    Returns:
        returns: 할인된 수익
    """
    returns = []
    R = 0
    
    # 역순으로 계산 (마지막 보상부터 시작)
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    
    # 텐서로 변환
    return torch.tensor(returns, device=rewards[0].device)

class WarmupScheduler:
    """워밍업 학습률 스케줄러 - 보다 유연한 API"""
    def __init__(self, 
                 optimizer: optim.Optimizer, 
                 warmup_steps: int, 
                 scheduler: Optional[Any] = None):
        """
        Args:
            optimizer: 옵티마이저
            warmup_steps: 워밍업 스텝 수
            scheduler: 워밍업 후 사용할 스케줄러
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.scheduler = scheduler
        self.current_step = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # 처음에는 워밍업 단계
        self.in_warmup = True
    
    def step(self) -> None:
        """스케줄러 스텝 진행"""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # 워밍업 단계: 선형적으로 학습률 증가
            lr_scale = min(1.0, self.current_step / self.warmup_steps)
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * lr_scale
            self.in_warmup = True
        else:
            # 워밍업 이후: 기본 스케줄러 사용
            if self.scheduler:
                self.scheduler.step()
            self.in_warmup = False
    
    def get_last_lr(self) -> List[float]:
        """현재 학습률 반환"""
        if self.in_warmup or not self.scheduler:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            return self.scheduler.get_last_lr()

class EarlyStopping:
    """조기 종료 - 검증 성능 기반"""
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Args:
            patience: 개선 없이 기다릴 에포크 수
            min_delta: 개선으로 간주할 최소 변화량
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score: float) -> bool:
        """검증 점수 기반 조기 종료 여부 확인
        
        Args:
            val_score: 검증 점수 (높을수록 좋음)
        
        Returns:
            stop: 종료 여부
        """
        if self.best_score is None:
            # 첫 번째 호출
            self.best_score = val_score
        elif val_score <= self.best_score + self.min_delta:
            # 개선이 없거나 충분하지 않음
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 충분한 개선
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop

class AdaptiveClipper:
    """적응형 그래디언트 클리핑"""
    def __init__(self, 
                 initial_max_norm: float = 1.0, 
                 adapt_factor: float = 0.01,
                 min_max_norm: float = 0.1, 
                 max_max_norm: float = 10.0):
        """
        Args:
            initial_max_norm: 초기 최대 노름
            adapt_factor: 적응 속도
            min_max_norm: 최소 최대 노름
            max_max_norm: 최대 최대 노름
        """
        self.max_norm = initial_max_norm
        self.adapt_factor = adapt_factor
        self.min_max_norm = min_max_norm
        self.max_max_norm = max_max_norm
        self.last_grad_norm = None
    
    def clip_gradients(self, parameters: Any) -> float:
        """그래디언트 클리핑
        
        Args:
            parameters: 모델 파라미터
        
        Returns:
            grad_norm: 클리핑 전 그래디언트 노름
        """
        # 그래디언트 노름 계산
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_norm)
        
        # 적응형 최대 노름 업데이트
        if self.last_grad_norm is not None:
            # 그래디언트 노름 변화에 따라 최대 노름 조정
            if grad_norm > self.last_grad_norm * 1.5:
                # 그래디언트가 급증하면 최대 노름 증가
                self.max_norm = min(
                    self.max_norm * (1 + self.adapt_factor),
                    self.max_max_norm
                )
            elif grad_norm < self.last_grad_norm * 0.5:
                # 그래디언트가 급감하면 최대 노름 감소
                self.max_norm = max(
                    self.max_norm * (1 - self.adapt_factor),
                    self.min_max_norm
                )
        
        self.last_grad_norm = grad_norm
        return grad_norm

def create_optimizer(network: torch.nn.Module, 
                    lr: float, 
                    weight_decay: float = 1e-4,
                    optimizer_type: str = 'adam') -> optim.Optimizer:
    """옵티마이저 생성
    
    Args:
        network: 신경망
        lr: 학습률
        weight_decay: 가중치 감쇠
        optimizer_type: 옵티마이저 타입 (adam, adamw, sgd)
    
    Returns:
        optimizer: 옵티마이저
    """
    # L2 정규화를 적용할 파라미터만 선택
    decay_params = []
    no_decay_params = []
    
    for name, param in network.named_parameters():
        if 'bias' in name or 'norm' in name or 'embedding' in name:
            # 바이어스, 정규화 레이어, 임베딩은 정규화 제외
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    # 파라미터 그룹 구성
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    # 옵티마이저 생성
    if optimizer_type.lower() == 'adam':
        return optim.Adam(param_groups, lr=lr)
    elif optimizer_type.lower() == 'adamw':
        return optim.AdamW(param_groups, lr=lr)
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(param_groups, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"지원되지 않는 옵티마이저 타입: {optimizer_type}")

def create_lr_scheduler(
        optimizer: optim.Optimizer, 
        num_epochs: int,
        warmup_steps: Optional[int] = None,
        scheduler_type: str = 'cosine'
    ) -> Any:
    """학습률 스케줄러 생성
    
    Args:
        optimizer: 옵티마이저
        num_epochs: 총 에포크 수
        warmup_steps: 워밍업 스텝 수
        scheduler_type: 스케줄러 타입 (cosine, step, linear)
    
    Returns:
        scheduler: 학습률 스케줄러
    """
    # 기본 스케줄러 선택
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=num_epochs // 10, gamma=0.1
        )
    elif scheduler_type == 'linear':
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.01, total_iters=num_epochs
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=num_epochs // 4, gamma=0.1
        )
    
    # 워밍업 래퍼
    if warmup_steps:
        scheduler = WarmupScheduler(optimizer, warmup_steps, scheduler)
    
    return scheduler

def train_model(
        env: 'MissionEnvironment', 
        val_env: 'MissionEnvironment', 
        policy_net: 'TransformerActorCriticNetwork',
        optimizer_actor: optim.Optimizer, 
        optimizer_critic: optim.Optimizer, 
        scheduler_actor: Any, 
        scheduler_critic: Any,
        device: torch.device, 
        edge_indices_cache: Dict[int, torch.Tensor], 
        config: 'TrainingConfig',
        checkpoint_path: Optional[str] = None, 
        results_dir: str = "./results",
        run_name: Optional[str] = None
    ) -> Dict[str, Any]:
    """모델 학습 - WandB 및 tqdm 진행도 강화
    
    Args:
        env: 훈련 환경
        val_env: 검증 환경
        policy_net: 정책 네트워크
        optimizer_actor: 액터 옵티마이저
        optimizer_critic: 크리틱 옵티마이저
        scheduler_actor: 액터 학습률 스케줄러
        scheduler_critic: 크리틱 학습률 스케줄러
        device: 계산 장치
        edge_indices_cache: 엣지 인덱스 캐시
        config: 학습 설정
        checkpoint_path: 체크포인트 경로
        results_dir: 결과 저장 디렉토리
        run_name: 실행 이름
    
    Returns:
        training_stats: 학습 통계
    """
    # 결과 디렉토리 설정
    os.makedirs(results_dir, exist_ok=True)
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # 실행 이름 설정
    if run_name is None:
        run_name = f"multi_uav_mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # WandB 설정 및 초기화
    if config.use_wandb:
        # WandB 설정
        config.wandb_config.name = run_name
        config.wandb_config.job_type = "training"
        config.wandb_config.tags = [
            f"uavs_{env.num_uavs}", 
            f"missions_{env.num_missions}",
            "transformer_model",
            "curriculum" if config.use_curriculum else "no_curriculum"
        ]
        
        # 네트워크 구조 정보 추가
        model_config = {
            "model": {
                "hidden_dim": config.hidden_dim,
                "num_gat_layers": config.num_gat_layers,
                "gat_heads": config.gat_heads,
                "transformer_layers": config.transformer_num_layers,
                "dropout": config.dropout,
                "uavs": env.num_uavs,
                "missions": env.max_missions,
                "params": sum(p.numel() for p in policy_net.parameters())
            }
        }
        
        # WandB 초기화
        wandb_run = config.wandb_config.initialize({**vars(config), **model_config})
        
        # 모델 그래프 로깅 (PyTorch 모델)
        if config.wandb_config.log_model:
            wandb.watch(policy_net, log="all", log_freq=100)
        
        # 환경 설정 로깅
        env_config = {
            "environment": {
                "num_uavs": env.num_uavs,
                "num_missions": env.num_missions,
                "max_missions": env.max_missions,
                "risk_areas": env.risk_centers.shape[0],
                "no_entry_zones": env.zone_centers.shape[0]
            }
        }
        wandb.config.update(env_config)
    
    # 그래디언트 클리퍼 초기화
    grad_clipper = AdaptiveClipper(initial_max_norm=config.gradient_clip)
    
    # 학습 매개변수 초기화
    temperature = config.temperature
    best_reward = -float('inf')
    no_improvement_count = 0
    global_step = 0
    
    # 조기 종료 설정
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)
    
    # 학습 통계 추적
    stats = {
        "epoch_rewards": [],
        "val_rewards": [],
        "epoch_losses": [],
        "temperatures": [],
        "learning_rates": [],
        "best_model_epoch": 0,
        "best_reward": -float('inf'),
        # 추가 통계
        "policy_losses": [],
        "value_losses": [],
        "entropies": [],
        "grad_norms": [],
        "success_rates": []
    }
    
    # 체크포인트 로드 (있는 경우)
    start_epoch = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            print(f"체크포인트 로드 중: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 모델 및 옵티마이저 상태 로드
            policy_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
            optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
            
            # 학습 상태 로드
            start_epoch = checkpoint.get('epoch', 0) + 1
            temperature = checkpoint.get('temperature', temperature)
            best_reward = checkpoint.get('best_reward', -float('inf'))
            global_step = checkpoint.get('global_step', 0)
            
            print(f"체크포인트 로드 완료 (에포크 {start_epoch-1}부터 계속)")
            
            # WandB에 체크포인트 정보 로깅
            if config.use_wandb:
                wandb.config.update({"resume_checkpoint": checkpoint_path})
                wandb.log({
                    "resumed_from_epoch": start_epoch-1,
                    "resumed_global_step": global_step,
                    "resumed_best_reward": best_reward
                })
        except Exception as e:
            print(f"체크포인트 로드 오류: {e}")
            print("새로 시작합니다.")
    
    # 학습 시작
    print(f"학습 시작 (총 {config.num_epochs} 에포크)")
    total_start_time = time.time()
    
    # tqdm 에포크 진행바
    epoch_iterator = trange(
        start_epoch, config.num_epochs + 1, 
        desc="에포크", 
        unit="epoch",
        position=config.tqdm_position,
        leave=True,
        dynamic_ncols=True
    ) if config.use_tqdm else range(start_epoch, config.num_epochs + 1)
    
    try:
        for epoch in epoch_iterator:
            epoch_start_time = time.time()
            
            # 커리큘럼 에포크 업데이트
            if config.use_curriculum:
                env.curriculum_epoch = epoch
                val_env.curriculum_epoch = epoch
            
            # 에포크별 통계 초기화
            epoch_losses = []
            epoch_rewards = []
            epoch_policy_losses = []
            epoch_value_losses = []
            epoch_entropies = []
            episode_success_count = 0
            
            # 배치 단위 학습
            # tqdm 배치 진행바
            batch_iterator = tqdm(
                range(config.batch_size),
                desc=f"에포크 {epoch}/{config.num_epochs}",
                unit="batch",
                position=config.tqdm_position+1,
                leave=False,
                dynamic_ncols=True
            ) if config.use_tqdm else range(config.batch_size)
            
            for batch_idx in batch_iterator:
                # 환경 초기화
                state = env.reset(clone=False)  # 학습 중에는 불필요한 클론 제거
                done = False
                log_probs = []
                values = []
                rewards = []
                entropies = []
                
                # 현재 미션 수에 맞는 엣지 인덱스 가져오기
                num_missions = env.num_missions
                if num_missions not in edge_indices_cache:
                    edge_indices_cache[num_missions] = create_edge_index(num_missions).to(device)
                edge_index = edge_indices_cache[num_missions]
                batch = torch.zeros(num_missions, dtype=torch.long, device=device)
                
                # 에피소드 실행
                step_count = 0
                while not done and step_count < config.max_step_limit:
                    step_count += 1
                    global_step += 1
                    
                    # 액션 마스크 생성
                    action_mask = env.create_action_mask()
                    
                    # 정책 및 가치 계산
                    with torch.no_grad():  # 학습 단계에서 그래디언트 계산 방지
                        action_logits, state_values = policy_net(
                            env.missions, edge_index, batch, 
                            state['positions'], env.speeds, env.uav_types,
                            action_mask, env.assigned_missions
                        )
                    
                    # 액션 선택
                    actions = choose_action(action_logits, temperature, action_mask)
                    
                    # 로그 확률 계산 (유효한 액션에 대해서만)
                    episode_log_probs = []
                    episode_values = []
                    episode_entropies = []
                    
                    for i, action in enumerate(actions):
                        if action != -1:
                            # 로그 확률 계산
                            probs = F.softmax(action_logits[i], dim=-1)
                            log_prob = torch.log(probs[action] + 1e-10)
                            episode_log_probs.append(log_prob)
                            episode_values.append(state_values[i])
                            
                            # 엔트로피 계산
                            entropy = -(probs * torch.log(probs + 1e-10)).sum()
                            episode_entropies.append(entropy)
                    
                    # 다음 단계로 진행
                    next_state, step_rewards, done, info = env.step(actions)
                    
                    # 스텝별 보상 기록
                    if done:
                        # 에피소드가 완료된 경우 최종 보상
                        reward = compute_episode_reward(env, config)
                        
                        # 성공 여부 기록
                        if info.get('success', False):
                            episode_success_count += 1
                    else:
                        # 계속 진행 중인 경우 중간 보상 (작은 시간 페널티)
                        reward = -0.01 * step_rewards.sum()
                    
                    rewards.append(reward)
                    
                    # 로그 확률 및 가치 저장
                    if episode_log_probs:
                        log_probs.extend(episode_log_probs)
                        values.extend(episode_values)
                        entropies.extend(episode_entropies)
                    
                    # 상태 업데이트
                    state = next_state
                
                # 로그 확률이 없는 경우 (유효한 액션이 없었던 경우) 건너뛰기
                if not log_probs:
                    continue
                
                # 에피소드 총 보상
                R = torch.stack(rewards).sum()
                epoch_rewards.append(R.item())
                
                # 학습 준비
                values = torch.cat(values)
                log_probs = torch.stack(log_probs)
                entropies = torch.stack(entropies)
                
                # 기대 수익 계산
                returns = torch.tensor([R] * len(values), device=device, dtype=torch.float)
                
                # 어드밴티지 계산
                advantage = returns - values.detach()
                
                # 어드밴티지 정규화 (안정성 향상)
                if advantage.shape[0] > 1:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                
                # 정책 손실 계산 (REINFORCE with baseline)
                policy_loss = torch.stack([-lp * adv for lp, adv in zip(log_probs, advantage)]).mean()
                
                # 가치 손실 계산 (MSE)
                value_loss = F.mse_loss(values.squeeze(), returns)
                
                # 엔트로피 계산 (탐색 장려)
                entropy = entropies.mean()
                
                # 총 손실 계산
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # 통계 기록
                epoch_losses.append(loss.item())
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropies.append(entropy.item())
                
                # 역전파 및 옵티마이저 스텝
                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                loss.backward()
                
                # 그래디언트 클리핑
                grad_norm = grad_clipper.clip_gradients(policy_net.parameters())
                
                # 파라미터 업데이트
                optimizer_actor.step()
                optimizer_critic.step()
                
                # 배치 진행바 업데이트
                if config.use_tqdm:
                    batch_metrics = {
                        'loss': f"{loss.item():.4f}",
                        'reward': f"{R.item():.2f}",
                        'temp': f"{temperature:.2f}"
                    }
                    batch_iterator.set_postfix(batch_metrics)
                
                # 일정 간격으로 WandB 로깅
                if config.use_wandb and global_step % 10 == 0:
                    metrics = {
                        "global_step": global_step,
                        "step_loss": loss.item(),
                        "step_policy_loss": policy_loss.item(),
                        "step_value_loss": value_loss.item(),
                        "step_reward": R.item(),
                        "step_entropy": entropy.item(),
                        "temperature": temperature,
                        "grad_norm": grad_norm,
                        "learning_rate_actor": scheduler_actor.get_last_lr()[0],
                        "learning_rate_critic": scheduler_critic.get_last_lr()[0],
                        "current_curriculum_level": env.curriculum_difficulty if hasattr(env, 'curriculum_difficulty') else 0
                    }
                    wandb.log(metrics)
            
            # 온도 감소 (탐색 감소)
            temperature = max(temperature * config.temperature_decay, config.temperature_min)
            
            # 스케줄러 스텝
            scheduler_actor.step()
            scheduler_critic.step()
            
            # 에포크 평균 통계 계산
            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            avg_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
            avg_policy_loss = sum(epoch_policy_losses) / max(len(epoch_policy_losses), 1)
            avg_value_loss = sum(epoch_value_losses) / max(len(epoch_value_losses), 1)
            avg_entropy = sum(epoch_entropies) / max(len(epoch_entropies), 1)
            success_rate = episode_success_count / config.batch_size
            
            # 통계 저장
            stats["epoch_rewards"].append(avg_reward)
            stats["epoch_losses"].append(avg_loss)
            stats["policy_losses"].append(avg_policy_loss)
            stats["value_losses"].append(avg_value_loss)
            stats["entropies"].append(avg_entropy)
            stats["temperatures"].append(temperature)
            stats["learning_rates"].append(scheduler_actor.get_last_lr()[0])
            stats["success_rates"].append(success_rate)
            
            # 에포크 경과 시간
            epoch_time = time.time() - epoch_start_time
            
            # 에포크 진행바 업데이트
            if config.use_tqdm:
                epoch_metrics = {
                    'loss': f"{avg_loss:.4f}",
                    'reward': f"{avg_reward:.2f}",
                    'success': f"{success_rate*100:.1f}%",
                    'temp': f"{temperature:.2f}",
                    'time': f"{epoch_time:.1f}s"
                }
                if isinstance(epoch_iterator, tqdm):
                    epoch_iterator.set_postfix(epoch_metrics)
            
            # 실시간 그래프 (설정된 경우)
            if config.live_plot and epoch % 5 == 0:
                plot_metrics = {
                    "Rewards": stats["epoch_rewards"],
                    "Losses": stats["epoch_losses"],
                    "Policy Losses": stats["policy_losses"],
                    "Value Losses": stats["value_losses"],
                    "Success Rates": stats["success_rates"]
                }
                plot_progress(plot_metrics, title=f"Training Progress (Epoch {epoch})")
            
            # 검증 실행
            if epoch % config.validation_interval == 0:
                val_rewards = []
                success_count = 0
                
                print(f"\n[에포크 {epoch}] 검증 중...")
                
                # 여러 검증 에피소드 실행
                val_iterator = tqdm(
                    range(5),
                    desc="검증",
                    position=config.tqdm_position+1,
                    leave=False
                ) if config.use_tqdm else range(5)
                
                for val_idx in val_iterator:
                    val_state = val_env.reset()
                    val_done = False
                    
                    # 엣지 인덱스 준비
                    val_num_missions = val_env.num_missions
                    if val_num_missions not in edge_indices_cache:
                        edge_indices_cache[val_num_missions] = create_edge_index(val_num_missions).to(device)
                    val_edge_index = edge_indices_cache[val_num_missions]
                    val_batch = torch.zeros(val_num_missions, dtype=torch.long, device=device)
                    
                    # 결정적 정책으로 평가
                    val_step_count = 0
                    while not val_done and val_step_count < config.max_step_limit:
                        val_step_count += 1
                        
                        # 액션 마스크 생성
                        val_action_mask = val_env.create_action_mask()
                        
                        # 정책 및 가치 계산
                        with torch.no_grad():
                            val_action_logits, _ = policy_net(
                                val_env.missions, val_edge_index, val_batch,
                                val_state['positions'], val_env.speeds, val_env.uav_types,
                                val_action_mask, val_env.assigned_missions
                            )
                        
                        # 결정적 액션 선택
                        val_actions = choose_action(
                            val_action_logits, 0.01, val_action_mask, deterministic=True
                        )
                        
                        # 환경 진행
                        val_state, _, val_done, val_info = val_env.step(val_actions)
                    
                    # 에피소드 보상 계산
                    val_reward = compute_episode_reward(val_env, config).item()
                    val_rewards.append(val_reward)
                    
                    # 성공 여부 확인
                    if val_info.get('success', False):
                        success_count += 1
                        
                    # 검증 진행바 업데이트
                    if config.use_tqdm:
                        val_metrics = {
                            'reward': f"{val_reward:.2f}",
                            'success': f"{val_info.get('success', False)}"
                        }
                        if isinstance(val_iterator, tqdm):
                            val_iterator.set_postfix(val_metrics)
                
                # 평균 검증 보상 계산
                avg_val_reward = sum(val_rewards) / len(val_rewards)
                val_success_rate = success_count / 5
                
                # 검증 통계 저장
                stats["val_rewards"].append(avg_val_reward)
                
                # 커리큘럼 성공률 추적
                if config.use_curriculum:
                    config.success_rates.append(val_success_rate)
                
                # WandB 로깅 - 검증 결과
                if config.use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "val_reward": avg_val_reward,
                        "val_success_rate": val_success_rate,
                        "curriculum_level": val_env.curriculum_difficulty if hasattr(val_env, 'curriculum_difficulty') else val_env.num_missions
                    })
                
                # 베스트 모델 저장
                if avg_val_reward > best_reward:
                    best_reward = avg_val_reward
                    stats["best_reward"] = best_reward
                    stats["best_model_epoch"] = epoch
                    no_improvement_count = 0
                    
                    # 베스트 모델 저장
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': policy_net.state_dict(),
                        'optimizer_actor_state_dict': optimizer_actor.state_dict(),
                        'optimizer_critic_state_dict': optimizer_critic.state_dict(),
                        'temperature': temperature,
                        'best_reward': best_reward,
                        'global_step': global_step,
                        'config': config
                    }, os.path.join(checkpoints_dir, "best_model.pth"))
                    
                    # WandB에 베스트 모델 로깅
                    if config.use_wandb and config.wandb_config.log_model:
                        wandb.save(os.path.join(checkpoints_dir, "best_model.pth"))
                    
                    print(f"[에포크 {epoch}] 최고 보상 갱신: {best_reward:.2f} (성공률: {val_success_rate:.2f})")
                else:
                    no_improvement_count += 1
                    print(f"[에포크 {epoch}] 보상 {avg_val_reward:.2f}, 개선 없음: {no_improvement_count} (성공률: {val_success_rate:.2f})")
            
            # 정기적인 체크포인트 저장
            if epoch % config.checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoints_dir, f"model_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': policy_net.state_dict(),
                    'optimizer_actor_state_dict': optimizer_actor.state_dict(),
                    'optimizer_critic_state_dict': optimizer_critic.state_dict(),
                    'temperature': temperature,
                    'best_reward': best_reward,
                    'global_step': global_step,
                    'config': config
                }, checkpoint_path)
                
                # WandB에 체크포인트 저장
                if config.use_wandb and config.wandb_config.log_model:
                    wandb.save(checkpoint_path)
                
                print(f"[에포크 {epoch}] 체크포인트 저장: {checkpoint_path}")
            
            # WandB 로깅 - 에포크 요약
            if config.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "loss": avg_loss,
                    "reward": avg_reward,
                    "policy_loss": avg_policy_loss,
                    "value_loss": avg_value_loss,
                    "entropy": avg_entropy,
                    "temperature": temperature,
                    "learning_rate_actor": scheduler_actor.get_last_lr()[0],
                    "learning_rate_critic": scheduler_critic.get_last_lr()[0],
                    "success_rate": success_rate,
                    "no_improvement_count": no_improvement_count,
                    "best_reward": best_reward,
                    "epoch_time": epoch_time
                })
            
            # 조기 종료 확인
            if epoch % config.validation_interval == 0:
                should_stop = early_stopping(avg_val_reward)
                if should_stop:
                    print(f"조기 종료: {config.early_stopping_patience} 에포크 동안 개선 없음")
                    break
        
        # 총 학습 시간
        total_time = time.time() - total_start_time
        
        # 최종 통계
        final_stats = {
            "total_time": total_time,
            "best_reward": best_reward,
            "best_epoch": stats["best_model_epoch"],
            "epochs_trained": epoch - start_epoch + 1,
            "global_steps": global_step
        }
        
        # 최종 통계 저장
        stats.update(final_stats)
        
        # WandB 로깅 - 최종 요약
        if config.use_wandb:
            wandb.run.summary.update({
                "best_reward": best_reward,
                "best_epoch": stats["best_model_epoch"],
                "total_training_time": total_time,
                "total_epochs": epoch - start_epoch + 1,
                "early_stopped": early_stopping.early_stop
            })
            
            # 최종 모델 성능 테이블
            model_performance = wandb.Table(
                columns=["Metric", "Value"],
                data=[
                    ["Best Reward", best_reward],
                    ["Best Epoch", stats["best_model_epoch"]],
                    ["Final Success Rate", stats["success_rates"][-1]],
                    ["Total Training Time (s)", total_time],
                    ["Total Epochs", epoch - start_epoch + 1],
                    ["Total Steps", global_step],
                    ["Early Stopped", early_stopping.early_stop]
                ]
            )
            wandb.log({"model_performance_summary": model_performance})
            
            # 학습 곡선 차트
            epoch_range = list(range(start_epoch, epoch+1))
            
            # 보상/손실 차트
            for metric_name, values in [
                ("rewards", stats["epoch_rewards"]),
                ("losses", stats["epoch_losses"]),
                ("policy_losses", stats["policy_losses"]),
                ("value_losses", stats["value_losses"]),
                ("entropies", stats["entropies"]),
                ("success_rates", stats["success_rates"])
            ]:
                if len(values) > 0:
                    table = wandb.Table(
                        columns=["epoch", metric_name],
                        data=[[e, v] for e, v in zip(epoch_range, values)]
                    )
                    wandb.log({
                        f"{metric_name}_chart": wandb.plot.line(
                            table, "epoch", metric_name,
                            title=f"Training {metric_name.replace('_', ' ').title()}"
                        )
                    })
            
            # 검증 보상 차트 (있는 경우)
            if len(stats["val_rewards"]) > 0:
                val_epochs = list(range(
                    start_epoch + config.validation_interval - 1, 
                    epoch + 1, 
                    config.validation_interval
                ))[:len(stats["val_rewards"])]
                
                val_table = wandb.Table(
                    columns=["epoch", "val_reward"],
                    data=[[e, v] for e, v in zip(val_epochs, stats["val_rewards"])]
                )
                wandb.log({
                    "val_rewards_chart": wandb.plot.line(
                        val_table, "epoch", "val_reward",
                        title="Validation Rewards"
                    )
                })
            
            # 학습 완료
            wandb.finish()
        
        print(f"학습 완료. 총 {epoch-start_epoch+1} 에포크, {total_time:.1f}초 소요.")
        print(f"최고 보상: {best_reward:.2f} (에포크 {stats['best_model_epoch']})")
        
        return stats
    
    except KeyboardInterrupt:
        print("\n학습 중단됨. 현재까지의 결과를 저장합니다.")
        
        # 중간 결과 저장
        interrupt_checkpoint_path = os.path.join(checkpoints_dir, "interrupted_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_actor_state_dict': optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': optimizer_critic.state_dict(),
            'temperature': temperature,
            'best_reward': best_reward,
            'global_step': global_step,
            'config': config
        }, interrupt_checkpoint_path)
        
        print(f"중단된 모델 저장됨: {interrupt_checkpoint_path}")
        
        # WandB에 중단 정보 기록
        if config.use_wandb:
            wandb.run.summary.update({
                "interrupted": True,
                "interrupted_epoch": epoch,
                "best_reward_before_interrupt": best_reward
            })
            wandb.finish()
        
        return stats

def evaluate_model(
        model: 'TransformerActorCriticNetwork',
        env: 'MissionEnvironment',
        edge_indices_cache: Dict[int, torch.Tensor],
        device: torch.device,
        config: Optional['TrainingConfig'] = None,
        num_episodes: int = 10,
        render: bool = False,
        render_path: Optional[str] = None,
        deterministic: bool = True,
        temperature: float = 0.01,
        use_wandb: bool = False,
        wandb_prefix: str = "eval"
    ) -> Dict[str, Any]:
    """학습된 모델 평가 - WandB 및 tqdm 진행도 표시 향상
    
    Args:
        model: 평가할 모델
        env: 평가 환경
        edge_indices_cache: 엣지 인덱스 캐시
        device: 계산 장치
        config: 학습 설정 (옵션)
        num_episodes: 평가 에피소드 수
        render: 시각화 여부
        render_path: 시각화 이미지 저장 경로
        deterministic: 결정적 정책 사용 여부
        temperature: 샘플링 온도
        use_wandb: WandB 사용 여부
        wandb_prefix: WandB 로깅 접두사
    
    Returns:
        metrics: 평가 지표
    """
    # 평가 모드로 설정
    model.eval()
    
    # tqdm 사용 여부 확인
    use_tqdm = True
    if config is not None:
        use_tqdm = config.use_tqdm
    
    # 평가 통계 초기화
    rewards = []
    success_count = 0
    episode_lengths = []
    travel_times = []
    mission_completions = []
    uav_travel_times = []
    episode_data = []
    
    # 평가 설명 출력
    print(f"\n{'='*50}")
    print(f"모델 평가 시작 (에피소드 {num_episodes}개)")
    print(f"{'='*50}")
    
    # WandB 초기화 (필요한 경우)
    if use_wandb and not wandb.run:
        wandb.init(project="multi_uav_mission", name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # 평가 시작 시간
    start_time = time.time()
    
    # tqdm 평가 진행바
    eval_iterator = tqdm(
        range(num_episodes),
        desc="평가 진행도",
        unit="episodes",
        dynamic_ncols=True,
        position=0
    ) if use_tqdm else range(num_episodes)
    
    # 여러 에피소드 평가
    for episode in eval_iterator:
        episode_start_time = time.time()
        
        # 환경 초기화
        state = env.reset()
        done = False
        
        # 엣지 인덱스 준비
        num_missions = env.num_missions
        if num_missions not in edge_indices_cache:
            edge_indices_cache[num_missions] = create_edge_index(num_missions).to(device)
        edge_index = edge_indices_cache[num_missions]
        batch = torch.zeros(num_missions, dtype=torch.long, device=device)
        
        # 에피소드 실행
        step_count = 0
        
        # 진행 상황 추적을 위한 tqdm 스텝 진행바
        step_iterator = tqdm(
            total=100,  # 최대 100 스텝 (예상)
            desc=f"에피소드 {episode+1}/{num_episodes}",
            unit="steps",
            leave=False,
            position=1,
            dynamic_ncols=True
        ) if use_tqdm else None
        
        while not done and step_count < 100:  # 최대 100 스텝
            step_count += 1
            
            # 액션 마스크 생성
            action_mask = env.create_action_mask()
            
            # 정책 예측
            with torch.no_grad():
                action_logits, _ = model(
                    env.missions, edge_index, batch,
                    state['positions'], env.speeds, env.uav_types,
                    action_mask, env.assigned_missions
                )
            
            # 액션 선택
            actions = choose_action(
                action_logits, temperature, action_mask, deterministic=deterministic
            )
            
            # 환경 진행
            state, step_rewards, done, info = env.step(actions)
            
            # 스텝 진행바 업데이트
            if step_iterator:
                # 미션 완료 비율 계산
                completion_ratio = env.visited.sum().item() / env.num_missions
                
                # 진행바 업데이트
                step_iterator.update(1)
                step_iterator.set_postfix({
                    'missions': f"{env.visited.sum().item()}/{env.num_missions}",
                    'completion': f"{completion_ratio*100:.1f}%",
                    'time': f"{env.cumulative_travel_times.sum().item():.1f}"
                })
        
        # 스텝 진행바 닫기
        if step_iterator:
            step_iterator.close()
        
        # 에피소드 종료 시간
        episode_time = time.time() - episode_start_time
        
        # 에피소드 통계 수집
        # 보상 계산 (config가 있으면 사용, 없으면 None 전달)
        reward = compute_episode_reward(env, config).item()
        rewards.append(reward)
        
        # 성공 여부 기록
        success = info.get('success', False)
        if success:
            success_count += 1
        
        # 에피소드 길이
        episode_lengths.append(step_count)
        
        # 이동 시간
        total_travel_time = env.cumulative_travel_times.sum().item()
        travel_times.append(total_travel_time)
        
        # UAV별 이동 시간
        uav_times = env.cumulative_travel_times.cpu().numpy().tolist()
        uav_travel_times.append(uav_times)
        
        # 완료된 미션 수
        completed_missions = env.visited.sum().item()
        mission_completions.append(completed_missions)
        
        # 에피소드 데이터 저장
        episode_data.append({
            'episode': episode,
            'success': success,
            'reward': reward,
            'steps': step_count,
            'travel_time': total_travel_time,
            'completed_missions': completed_missions,
            'uav_times': uav_times,
            'time_std': env.cumulative_travel_times.std().item() if env.num_uavs > 1 else 0,
            'processing_time': episode_time
        })
        
        # 시각화 (요청된 경우)
        if render:
            render_filename = None
            if render_path:
                os.makedirs(render_path, exist_ok=True)
                render_filename = os.path.join(render_path, f"episode_{episode+1}.png")
            
            # 경로 시각화
            env.visualize_paths(render_filename)
        
        # 에피소드 결과 로깅
        print(f"\n에피소드 {episode+1}: " + 
              f"{'성공' if success else '실패'} | " + 
              f"보상: {reward:.2f} | " + 
              f"스텝: {step_count} | " + 
              f"미션: {completed_missions}/{env.num_missions} | " + 
              f"시간: {total_travel_time:.2f}")
        
        # WandB 로깅 (에피소드별)
        if use_wandb:
            wandb.log({
                f"{wandb_prefix}_episode": episode,
                f"{wandb_prefix}_episode_reward": reward,
                f"{wandb_prefix}_episode_steps": step_count,
                f"{wandb_prefix}_episode_success": success,
                f"{wandb_prefix}_episode_travel_time": total_travel_time,
                f"{wandb_prefix}_episode_completed_missions": completed_missions,
                f"{wandb_prefix}_episode_completion_rate": completed_missions / env.num_missions,
                f"{wandb_prefix}_episode_time_std": env.cumulative_travel_times.std().item() if env.num_uavs > 1 else 0,
                f"{wandb_prefix}_episode_processing_time": episode_time
            })
            
            # 이미지 로깅 (렌더링된 경우)
            if render and render_path and os.path.exists(os.path.join(render_path, f"episode_{episode+1}.png")):
                wandb.log({
                    f"{wandb_prefix}_episode_path": wandb.Image(
                        os.path.join(render_path, f"episode_{episode+1}.png"),
                        caption=f"Episode {episode+1} Path"
                    )
                })
        
        # 진행바 업데이트
        if use_tqdm:
            eval_iterator.set_postfix({
                'reward': f"{reward:.2f}",
                'success': f"{success_count}/{episode+1}",
                'time': f"{episode_time:.2f}s"
            })
    
    # 총 평가 시간
    total_time = time.time() - start_time
    
    # 평가 지표 계산
    avg_reward = sum(rewards) / num_episodes
    success_rate = success_count / num_episodes
    avg_episode_length = sum(episode_lengths) / num_episodes
    avg_travel_time = sum(travel_times) / num_episodes
    avg_missions_completed = sum(mission_completions) / num_episodes
    
    # 표준 편차 계산
    if num_episodes > 1:
        std_reward = (sum((r - avg_reward) ** 2 for r in rewards) / num_episodes) ** 0.5
        std_episode_length = (sum((l - avg_episode_length) ** 2 for l in episode_lengths) / num_episodes) ** 0.5
        std_travel_time = (sum((t - avg_travel_time) ** 2 for t in travel_times) / num_episodes) ** 0.5
    else:
        std_reward = std_episode_length = std_travel_time = 0.0
    
    # 지표 딕셔너리 생성
    metrics = {
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "success_rate": success_rate,
        "avg_episode_length": avg_episode_length,
        "std_episode_length": std_episode_length,
        "avg_travel_time": avg_travel_time,
        "std_travel_time": std_travel_time,
        "avg_missions_completed": avg_missions_completed,
        "total_evaluation_time": total_time,
        # 상세 데이터
        "rewards": rewards,
        "episode_lengths": episode_lengths,
        "travel_times": travel_times,
        "mission_completions": mission_completions,
        "uav_travel_times": uav_travel_times,
        "episode_data": episode_data
    }
    
    # 요약 결과 출력
    print(f"\n{'='*50}")
    print(f"평가 결과 요약 ({num_episodes} 에피소드)")
    print(f"{'='*50}")
    print(f"평균 보상: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"성공률: {success_rate*100:.1f}% ({success_count}/{num_episodes})")
    print(f"평균 에피소드 길이: {avg_episode_length:.2f} ± {std_episode_length:.2f} 스텝")
    print(f"평균 이동 시간: {avg_travel_time:.2f} ± {std_travel_time:.2f}")
    print(f"평균 완료 미션: {avg_missions_completed:.2f}/{env.num_missions} ({avg_missions_completed/env.num_missions*100:.1f}%)")
    print(f"총 평가 시간: {total_time:.2f}초")
    print(f"{'='*50}")
    
    # WandB 로깅 (요약)
    if use_wandb:
        # 요약 지표
        wandb.log({
            f"{wandb_prefix}_summary_avg_reward": avg_reward,
            f"{wandb_prefix}_summary_std_reward": std_reward,
            f"{wandb_prefix}_summary_success_rate": success_rate,
            f"{wandb_prefix}_summary_avg_episode_length": avg_episode_length,
            f"{wandb_prefix}_summary_std_episode_length": std_episode_length,
            f"{wandb_prefix}_summary_avg_travel_time": avg_travel_time,
            f"{wandb_prefix}_summary_std_travel_time": std_travel_time,
            f"{wandb_prefix}_summary_avg_missions_completed": avg_missions_completed,
            f"{wandb_prefix}_summary_total_evaluation_time": total_time
        })
        
        # 히스토그램 및 박스 플롯
        for metric_name, values, title in [
            ("rewards", rewards, "Episode Rewards"),
            ("episode_lengths", episode_lengths, "Episode Lengths"),
            ("travel_times", travel_times, "Travel Times"),
            ("mission_completions", mission_completions, "Completed Missions")
        ]:
            # 히스토그램
            wandb.log({
                f"{wandb_prefix}_hist_{metric_name}": wandb.Histogram(values)
            })
            
            # 박스 플롯 데이터
            table = wandb.Table(columns=["metric", "value"])
            for v in values:
                table.add_data(metric_name, v)
            
            # 박스 플롯
            wandb.log({
                f"{wandb_prefix}_box_{metric_name}": wandb.plot.box(
                    table, "metric", "value",
                    title=title
                )
            })
        
        # 평가 요약 테이블
        summary_table = wandb.Table(
            columns=["Metric", "Value"],
            data=[
                ["Average Reward", f"{avg_reward:.4f} ± {std_reward:.4f}"],
                ["Success Rate", f"{success_rate*100:.1f}%"],
                ["Average Episode Length", f"{avg_episode_length:.2f} ± {std_episode_length:.2f}"],
                ["Average Travel Time", f"{avg_travel_time:.2f} ± {std_travel_time:.2f}"],
                ["Average Missions Completed", f"{avg_missions_completed:.2f}/{env.num_missions}"],
                ["Total Evaluation Time", f"{total_time:.2f}s"]
            ]
        )
        wandb.log({f"{wandb_prefix}_summary_table": summary_table})
        
        # 에피소드 상세 정보 테이블
        episodes_table = wandb.Table(
            columns=["Episode", "Success", "Reward", "Steps", "Travel Time", "Completed Missions", "Time Std"]
        )
        for data in episode_data:
            episodes_table.add_data(
                data['episode'] + 1,
                data['success'],
                data['reward'],
                data['steps'],
                data['travel_time'],
                data['completed_missions'],
                data['time_std']
            )
        wandb.log({f"{wandb_prefix}_episodes_table": episodes_table})
    
    return metrics

def interactive_evaluation(
        model: 'TransformerActorCriticNetwork',
        env: 'MissionEnvironment',
        edge_indices_cache: Dict[int, torch.Tensor],
        device: torch.device,
        config: Optional['TrainingConfig'] = None,
        render_path: Optional[str] = None,
        step_by_step: bool = True
    ) -> None:
    """대화형 평가 - 모델의 결정을 단계별로 시각화
    
    Args:
        model: 평가할 모델
        env: 평가 환경
        edge_indices_cache: 엣지 인덱스 캐시
        device: 계산 장치
        config: 학습 설정 (옵션)
        render_path: 시각화 이미지 저장 경로
        step_by_step: 단계별 평가 여부
    """
    # 평가 모드로 설정
    model.eval()
    
    # 렌더링 경로 설정
    if render_path:
        os.makedirs(render_path, exist_ok=True)
    
    print("\n===== 대화형 평가 시작 =====")
    print("각 단계에서 Enter를 누르면 진행, 'q'를 누르면 종료합니다.")
    
    input("평가를 시작하려면 Enter를 누르세요...")
    
    # 환경 초기화
    state = env.reset()
    env.visualize_paths(os.path.join(render_path, "initial_state.png") if render_path else None)
    
    # 단계별 진행
    done = False
    step = 0
    
    try:
        while not done:
            step += 1
            print(f"\n--- 스텝 {step} ---")
            
            # 미션 완료 상태
            completed = env.visited.sum().item()
            total = env.num_missions
            print(f"완료된 미션: {completed}/{total} ({completed/total*100:.1f}%)")
            
            # 이동 시간
            travel_time = env.cumulative_travel_times.sum().item()
            print(f"총 이동 시간: {travel_time:.2f}")
            
            # 엣지 인덱스 준비
            num_missions = env.num_missions
            if num_missions not in edge_indices_cache:
                edge_indices_cache[num_missions] = create_edge_index(num_missions).to(device)
            edge_index = edge_indices_cache[num_missions]
            batch = torch.zeros(num_missions, dtype=torch.long, device=device)
            
            # 액션 마스크 생성
            action_mask = env.create_action_mask()
            
            # 정책 예측
            with torch.no_grad():
                action_logits, state_values = model(
                    env.missions, edge_index, batch,
                    state['positions'], env.speeds, env.uav_types,
                    action_mask, env.assigned_missions
                )
            
            # 소프트맥스 확률 계산
            probs = F.softmax(action_logits, dim=-1)
            
            # 액션 선택
            actions = choose_action(action_logits, 0.01, action_mask, deterministic=True)
            
            # 액션 정보 출력
            print("\n선택된 액션:")
            for u, action in enumerate(actions):
                if action != -1:
                    # 주요 대안 액션 확인
                    valid_probs = probs[u].clone()
                    valid_probs[action_mask[u]] = 0.0
                    
                    # 상위 3개 액션 및 확률
                    top_actions = valid_probs.topk(min(3, (~action_mask[u]).sum().item()))
                    top_indices = top_actions.indices.cpu().tolist()
                    top_values = top_actions.values.cpu().tolist()
                    
                    # 선택된 액션 정보
                    print(f"UAV {u} -> 미션 {action} (확률: {probs[u][action]:.4f})")
                    
                    # 대안 액션 정보
                    alt_actions = [(idx, val) for idx, val in zip(top_indices, top_values) if idx != action]
                    if alt_actions:
                        print(f"  대안: " + ", ".join([f"미션 {idx} ({val:.4f})" for idx, val in alt_actions]))
                else:
                    print(f"UAV {u} -> 대기")
            
            # 상태 값 출력
            if state_values is not None:
                print(f"\n상태 값: {state_values.mean().item():.4f}")
            
            # 렌더링
            if render_path:
                env.visualize_paths(os.path.join(render_path, f"step_{step:03d}.png"))
            
            # 단계별 모드에서 사용자 입력 대기
            if step_by_step:
                user_input = input("\n다음 단계를 진행하려면 Enter, 종료하려면 'q'를 입력하세요: ")
                if user_input.lower() == 'q':
                    break
            
            # 환경 진행
            state, _, done, info = env.step(actions)
            
            # 종료 조건
            if done:
                print("\n===== 평가 완료 =====")
                print("모든 미션이 완료되었습니다!")
                
                # 최종 상태 출력
                reward = compute_episode_reward(env, config).item()
                print(f"최종 보상: {reward:.4f}")
                print(f"총 단계 수: {step}")
                print(f"총 이동 시간: {env.cumulative_travel_times.sum().item():.2f}")
                
                # UAV별 이동 시간
                for u in range(env.num_uavs):
                    print(f"UAV {u} 이동 시간: {env.cumulative_travel_times[u].item():.2f}")
                
                # 최종 상태 렌더링
                if render_path:
                    env.visualize_paths(os.path.join(render_path, "final_state.png"))
                    # 미션 할당 네트워크 시각화
                    visualize_mission_allocation(env, os.path.join(render_path, "mission_network.png"), "Final Mission Allocation")
            
            # 최대 단계 수 제한
            if step >= 100:
                print("\n===== 최대 단계 수 도달 =====")
                print("100단계에 도달하여 평가를 종료합니다.")
                break
    
    except KeyboardInterrupt:
        print("\n\n평가가 중단되었습니다.")
    
    print(f"\n평가 결과가 '{render_path}'에 저장되었습니다." if render_path else "")


def batch_evaluation(
        model_path: str,
        env_generator: Callable[[], 'MissionEnvironment'],
        edge_indices_cache: Dict[int, torch.Tensor],
        device: torch.device,
        model_class: type,
        model_args: Dict[str, Any],
        num_envs: int = 10,
        num_episodes_per_env: int = 5,
        results_dir: str = "./batch_eval_results",
        use_wandb: bool = False,
        use_tqdm: bool = True
    ) -> Dict[str, Any]:
    """배치 평가 - 여러 환경에서 모델 성능 평가
    
    Args:
        model_path: 모델 체크포인트 경로
        env_generator: 환경 생성 함수
        edge_indices_cache: 엣지 인덱스 캐시
        device: 계산 장치
        model_class: 모델 클래스
        model_args: 모델 생성 인자
        num_envs: 평가할 환경 수
        num_episodes_per_env: 환경당 평가할 에피소드 수
        results_dir: 결과 저장 디렉토리
        use_wandb: WandB 사용 여부
        use_tqdm: tqdm 사용 여부
    
    Returns:
        batch_metrics: 배치 평가 지표
    """
    # 결과 디렉토리 생성
    os.makedirs(results_dir, exist_ok=True)
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device)
    
    # 모델 생성
    model = model_class(**model_args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 학습 설정 (체크포인트에 있는 경우)
    config = checkpoint.get('config', None)
    
    # WandB 초기화
    if use_wandb:
        if not wandb.run:
            run_name = f"batch_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(project="multi_uav_mission", name=run_name, job_type="evaluation")
            
            # 체크포인트 정보 로깅
            wandb.config.update({
                "model_path": model_path,
                "num_envs": num_envs,
                "num_episodes_per_env": num_episodes_per_env,
                "checkpoint_epoch": checkpoint.get('epoch', 0)
            })
    
    # 결과 저장
    all_metrics = []
    
    # 환경별 평가
    env_iterator = trange(num_envs, desc="환경 평가", dynamic_ncols=True) if use_tqdm else range(num_envs)
    
    for env_idx in env_iterator:
        # 환경 생성
        env = env_generator()
        
        # 렌더링 경로
        render_path = os.path.join(results_dir, f"env_{env_idx}")
        
        # 환경 평가
        env_metrics = evaluate_model(
            model, env, edge_indices_cache, device, config,
            num_episodes=num_episodes_per_env,
            render=True,
            render_path=render_path,
            use_wandb=use_wandb,
            wandb_prefix=f"env_{env_idx}"
        )
        
        # 환경 정보 추가
        env_metrics['env_idx'] = env_idx
        env_metrics['num_uavs'] = env.num_uavs
        env_metrics['num_missions'] = env.num_missions
        env_metrics['risk_areas'] = env.risk_centers.shape[0]
        env_metrics['no_entry_zones'] = env.zone_centers.shape[0]
        
        # 결과 추가
        all_metrics.append(env_metrics)
        
        # 진행바 업데이트
        if use_tqdm:
            env_iterator.set_postfix({
                'success': f"{env_metrics['success_rate']*100:.1f}%",
                'reward': f"{env_metrics['avg_reward']:.2f}"
            })
        
        # WandB 로깅
        if use_wandb:
            wandb.log({
                f"env_{env_idx}_summary": {
                    "success_rate": env_metrics['success_rate'],
                    "avg_reward": env_metrics['avg_reward'],
                    "avg_travel_time": env_metrics['avg_travel_time'],
                    "num_uavs": env.num_uavs,
                    "num_missions": env.num_missions
                }
            })
    
    # 전체 결과 계산
    aggregated_metrics = {
        "overall_success_rate": np.mean([m['success_rate'] for m in all_metrics]),
        "overall_avg_reward": np.mean([m['avg_reward'] for m in all_metrics]),
        "overall_std_reward": np.std([m['avg_reward'] for m in all_metrics]),
        "overall_avg_travel_time": np.mean([m['avg_travel_time'] for m in all_metrics]),
        "overall_avg_steps": np.mean([m['avg_episode_length'] for m in all_metrics]),
        "min_reward": min([m['avg_reward'] for m in all_metrics]),
        "max_reward": max([m['avg_reward'] for m in all_metrics]),
        "min_success_rate": min([m['success_rate'] for m in all_metrics]),
        "max_success_rate": max([m['success_rate'] for m in all_metrics]),
        "all_env_metrics": all_metrics
    }
    
    # 결과 요약 출력
    print(f"\n{'='*50}")
    print(f"배치 평가 결과 요약 ({num_envs} 환경, 각 {num_episodes_per_env} 에피소드)")
    print(f"{'='*50}")
    print(f"전체 성공률: {aggregated_metrics['overall_success_rate']*100:.1f}%")
    print(f"전체 평균 보상: {aggregated_metrics['overall_avg_reward']:.2f} ± {aggregated_metrics['overall_std_reward']:.2f}")
    print(f"최소 보상: {aggregated_metrics['min_reward']:.2f}")
    print(f"최대 보상: {aggregated_metrics['max_reward']:.2f}")
    print(f"최소 성공률: {aggregated_metrics['min_success_rate']*100:.1f}%")
    print(f"최대 성공률: {aggregated_metrics['max_success_rate']*100:.1f}%")
    print(f"전체 평균 이동 시간: {aggregated_metrics['overall_avg_travel_time']:.2f}")
    print(f"전체 평균 스텝 수: {aggregated_metrics['overall_avg_steps']:.2f}")
    print(f"{'='*50}")
    
    # 결과 저장
    results_path = os.path.join(results_dir, "batch_eval_results.json")
    
    try:
        import json
        
        # 텐서 및 NumPy 배열을 JSON 직렬화 가능한 형태로 변환
        def serialize_for_json(obj):
            if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.cpu().tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(results_path, 'w') as f:
            json.dump(aggregated_metrics, f, default=serialize_for_json, indent=2)
        
        print(f"결과가 '{results_path}'에 저장되었습니다.")
    except Exception as e:
        print(f"결과 저장 중 오류 발생: {e}")
    
    # WandB 로깅
    if use_wandb:
        # 최종 요약
        wandb.log({
            "batch_overall_success_rate": aggregated_metrics['overall_success_rate'],
            "batch_overall_avg_reward": aggregated_metrics['overall_avg_reward'],
            "batch_overall_std_reward": aggregated_metrics['overall_std_reward'],
            "batch_min_reward": aggregated_metrics['min_reward'],
            "batch_max_reward": aggregated_metrics['max_reward'],
            "batch_min_success_rate": aggregated_metrics['min_success_rate'],
            "batch_max_success_rate": aggregated_metrics['max_success_rate']
        })
        
        # 요약 테이블
        summary_table = wandb.Table(
            columns=["Metric", "Value"],
            data=[
                ["Overall Success Rate", f"{aggregated_metrics['overall_success_rate']*100:.1f}%"],
                ["Overall Average Reward", f"{aggregated_metrics['overall_avg_reward']:.2f} ± {aggregated_metrics['overall_std_reward']:.2f}"],
                ["Min Reward", f"{aggregated_metrics['min_reward']:.2f}"],
                ["Max Reward", f"{aggregated_metrics['max_reward']:.2f}"],
                ["Min Success Rate", f"{aggregated_metrics['min_success_rate']*100:.1f}%"],
                ["Max Success Rate", f"{aggregated_metrics['max_success_rate']*100:.1f}%"]
            ]
        )
        wandb.log({"batch_summary_table": summary_table})
        
        # 환경별 결과 테이블
        env_table = wandb.Table(
            columns=[
                "Environment", "Success Rate", "Avg Reward", "Avg Travel Time", 
                "UAVs", "Missions", "Risk Areas", "No Entry Zones"
            ]
        )
        
        for m in all_metrics:
            env_table.add_data(
                m['env_idx'],
                f"{m['success_rate']*100:.1f}%",
                f"{m['avg_reward']:.2f}",
                f"{m['avg_travel_time']:.2f}",
                m['num_uavs'],
                m['num_missions'],
                m['risk_areas'],
                m['no_entry_zones']
            )
        
        wandb.log({"batch_env_results": env_table})
        
        # 환경별 성능 비교 차트
        env_indices = [m['env_idx'] for m in all_metrics]
        success_rates = [m['success_rate'] for m in all_metrics]
        rewards = [m['avg_reward'] for m in all_metrics]
        
        comparison_table = wandb.Table(
            columns=["env_idx", "success_rate", "reward"],
            data=[[e, s, r] for e, s, r in zip(env_indices, success_rates, rewards)]
        )
        
        wandb.log({
            "batch_env_success_comparison": wandb.plot.bar(
                comparison_table, "env_idx", "success_rate",
                title="Success Rate by Environment"
            ),
            "batch_env_reward_comparison": wandb.plot.bar(
                comparison_table, "env_idx", "reward",
                title="Reward by Environment"
            )
        })
        
        wandb.finish()
    
    return aggregated_metrics

def visualize_training_results(stats: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """학습 결과 시각화
    
    Args:
        stats: 학습 통계
        save_path: 이미지 저장 경로
    """
    try:
        import matplotlib.pyplot as plt
        
        # 그림 설정
        plt.figure(figsize=(15, 10))
        
        # 서브플롯 1: 에포크별 보상
        plt.subplot(2, 2, 1)
        plt.plot(stats["epoch_rewards"], label="Train Reward")
        if "val_rewards" in stats and stats["val_rewards"]:
            # 검증 간격에 맞춰 x 좌표 조정
            val_epochs = list(range(0, len(stats["epoch_rewards"]), 
                                   len(stats["epoch_rewards"]) // len(stats["val_rewards"]) + 1))[:len(stats["val_rewards"])]
            plt.plot(val_epochs, stats["val_rewards"], label="Validation Reward", linestyle='--', marker='o')
        plt.axhline(y=stats.get("best_reward", 0), color='r', linestyle='-', alpha=0.3, label="Best Reward")
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.title("Training and Validation Rewards")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 2: 에포크별 손실
        plt.subplot(2, 2, 2)
        plt.plot(stats["epoch_losses"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 3: 온도 감소
        plt.subplot(2, 2, 3)
        plt.plot(stats["temperatures"])
        plt.xlabel("Epoch")
        plt.ylabel("Temperature")
        plt.title("Temperature Decay")
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 4: 학습률
        plt.subplot(2, 2, 4)
        plt.plot(stats["learning_rates"])
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True, alpha=0.3)
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 저장 또는 표시
        if save_path:
            plt.savefig(save_path)
            print(f"학습 결과 시각화가 저장되었습니다: {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    except ImportError:
        print("matplotlib이 설치되어 있지 않아 시각화를 수행할 수 없습니다.")

def visualize_evaluation_metrics(metrics: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """평가 지표 시각화
    
    Args:
        metrics: 평가 지표
        save_path: 이미지 저장 경로
    """
    try:
        import matplotlib.pyplot as plt
        
        # 그림 설정
        plt.figure(figsize=(15, 10))
        
        # 서브플롯 1: 에피소드별 보상
        plt.subplot(2, 2, 1)
        plt.bar(range(len(metrics["rewards"])), metrics["rewards"])
        plt.axhline(y=metrics["avg_reward"], color='r', linestyle='-', 
                   label=f"Avg: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Rewards")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 2: 에피소드별 길이
        plt.subplot(2, 2, 2)
        plt.bar(range(len(metrics["episode_lengths"])), metrics["episode_lengths"])
        plt.axhline(y=metrics["avg_episode_length"], color='r', linestyle='-', 
                   label=f"Avg: {metrics['avg_episode_length']:.2f} ± {metrics['std_episode_length']:.2f}")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.title("Episode Lengths")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 3: 에피소드별 이동 시간
        plt.subplot(2, 2, 3)
        plt.bar(range(len(metrics["travel_times"])), metrics["travel_times"])
        plt.axhline(y=metrics["avg_travel_time"], color='r', linestyle='-', 
                   label=f"Avg: {metrics['avg_travel_time']:.2f} ± {metrics['std_travel_time']:.2f}")
        plt.xlabel("Episode")
        plt.ylabel("Travel Time")
        plt.title("Total Travel Times")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 4: 에피소드별 완료된 미션 수
        plt.subplot(2, 2, 4)
        plt.bar(range(len(metrics["mission_completions"])), metrics["mission_completions"])
        plt.axhline(y=metrics["avg_missions_completed"], color='r', linestyle='-', 
                   label=f"Avg: {metrics['avg_missions_completed']:.2f}")
        plt.xlabel("Episode")
        plt.ylabel("Completed Missions")
        plt.title("Mission Completions")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 추가 정보 표시
        plt.figtext(0.5, 0.01, f"Success Rate: {metrics['success_rate']*100:.1f}%", 
                   ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # 레이아웃 조정
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 저장 또는 표시
        if save_path:
            plt.savefig(save_path)
            print(f"평가 지표 시각화가 저장되었습니다: {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    except ImportError:
        print("matplotlib이 설치되어 있지 않아 시각화를 수행할 수 없습니다.")

def visualize_mission_allocation(
        env: 'MissionEnvironment',
        save_path: Optional[str] = None,
        title: str = "UAV Mission Allocation"
    ) -> None:
    """UAV 미션 할당 시각화
    
    Args:
        env: 미션 환경
        save_path: 이미지 저장 경로
        title: 그래프 제목
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        from matplotlib.patches import Circle
        
        # 그래프 생성
        G = nx.DiGraph()
        
        # 미션 노드 추가
        missions_np = env.missions.cpu().numpy()
        for i in range(env.missions.shape[0]):
            pos = missions_np[i]
            if i == 0:
                G.add_node(f"M{i}", pos=pos, color='green', label="Start")
            elif i == env.missions.shape[0] - 1:
                G.add_node(f"M{i}", pos=pos, color='red', label="End")
            else:
                G.add_node(f"M{i}", pos=pos, color='blue', label=f"M{i}")
        
        # UAV 노드 추가
        for u in range(env.num_uavs):
            uav_pos = env.current_positions[u].cpu().numpy()
            uav_type = "Fixed" if env.uav_types[u].item() == 0 else "Rotary"
            G.add_node(f"UAV{u}", pos=uav_pos, color='orange', 
                      label=f"UAV{u} ({uav_type})")
        
        # 미션 할당 엣지 추가
        for u, missions in enumerate(env.assigned_missions):
            if not missions:
                continue
            
            # UAV와 첫 미션 연결
            G.add_edge(f"UAV{u}", f"M{missions[0]}", color='orange')
            
            # 미션 간 연결
            for i in range(len(missions) - 1):
                G.add_edge(f"M{missions[i]}", f"M{missions[i+1]}", color='orange')
        
        # 그래프 그리기
        plt.figure(figsize=(12, 10))
        
        # 노드 위치
        pos = nx.get_node_attributes(G, 'pos')
        
        # 노드 색상
        node_colors = [data.get('color', 'blue') for _, data in G.nodes(data=True)]
        
        # 엣지 색상
        edge_colors = [data.get('color', 'black') for _, _, data in G.edges(data=True)]
        
        # 노드 그리기
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
        
        # 엣지 그리기
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, arrowsize=15)
        
        # 노드 레이블
        labels = {node: data.get('label', node) for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
        
        # 위험 지역 그리기
        ax = plt.gca()
        if env.risk_centers.shape[0] > 0:
            for i in range(env.risk_centers.shape[0]):
                center = env.risk_centers[i].cpu().numpy()
                radius = env.risk_radii[i].item()
                circle = Circle(center, radius, alpha=0.2, color='red', 
                               label="Risk Area" if i == 0 else "")
                ax.add_patch(circle)
        
        # 출입 불가 지역 그리기
        if env.zone_centers.shape[0] > 0:
            for i in range(env.zone_centers.shape[0]):
                center = env.zone_centers[i].cpu().numpy()
                radius = env.zone_radii[i].item()
                circle = Circle(center, radius, alpha=0.4, color='black', 
                               label="No Entry Zone" if i == 0 else "")
                ax.add_patch(circle)
        
        # 미션 완료 정보
        completion_info = f"Missions: {env.visited.sum().item()}/{env.num_missions}"
        travel_info = f"Total Travel Time: {env.cumulative_travel_times.sum().item():.1f}"
        
        # 범례 및 정보 표시
        plt.figtext(0.5, 0.01, f"{completion_info} | {travel_info}", 
                   ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        # 그래프 설정
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        
        # 저장 또는 표시
        if save_path:
            plt.savefig(save_path)
            print(f"미션 할당 시각화가 저장되었습니다: {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    except ImportError:
        print("matplotlib 또는 networkx가 설치되어 있지 않아 시각화를 수행할 수 없습니다.")
        
# 메인 함수 - 개선된 모니터링 및 진행도 표시
def main():
    """다중 UAV 미션 할당 시스템 메인 함수"""
    parser = argparse.ArgumentParser(description="다중 UAV 미션 할당 시스템")
    
    # 기본 설정
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'eval', 'demo', 'batch_eval', 'interactive'],
                       help='실행 모드 (train, eval, demo, batch_eval, interactive)')
    parser.add_argument('--num_uavs', type=int, default=3, help='UAV 수')
    parser.add_argument('--num_missions', type=int, default=20, help='미션 수')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--device', type=str, default=None, 
                        help='계산 장치 (cuda, cpu, None=자동 감지)')
    
    # 학습 관련 설정
    parser.add_argument('--num_epochs', type=int, default=100, help='학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--lr_actor', type=float, default=1e-4, help='액터 학습률')
    parser.add_argument('--lr_critic', type=float, default=1e-4, help='크리틱 학습률')
    parser.add_argument('--temperature', type=float, default=1.0, help='샘플링 온도')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='체크포인트 경로')
    parser.add_argument('--results_dir', type=str, default='./results', help='결과 저장 디렉토리')
    
    # 모델 아키텍처 관련 설정
    parser.add_argument('--hidden_dim', type=int, default=128, help='히든 레이어 차원')
    parser.add_argument('--num_gat_layers', type=int, default=3, help='GAT 레이어 수')
    parser.add_argument('--gat_heads', type=int, default=8, help='GAT 헤드 수')
    parser.add_argument('--transformer_layers', type=int, default=3, help='트랜스포머 레이어 수')
    parser.add_argument('--dropout', type=float, default=0.1, help='드롭아웃 비율')
    
    # 커리큘럼 학습 관련 설정
    parser.add_argument('--no_curriculum', action='store_true', help='커리큘럼 학습 비활성화')
    parser.add_argument('--min_missions', type=int, default=5, help='커리큘럼 최소 미션 수')
    parser.add_argument('--adaptive_curriculum', action='store_true', help='적응형 커리큘럼 사용')
    
    # 평가 관련 설정
    parser.add_argument('--eval_episodes', type=int, default=10, help='평가 에피소드 수')
    parser.add_argument('--render', action='store_true', help='시각화 활성화')
    parser.add_argument('--render_path', type=str, default=None, help='시각화 저장 경로')
    
    # 배치 평가 관련 설정
    parser.add_argument('--num_envs', type=int, default=10, help='배치 평가 환경 수')
    parser.add_argument('--episodes_per_env', type=int, default=5, help='환경당 에피소드 수')
    
    # WandB 관련 설정
    parser.add_argument('--no_wandb', action='store_true', help='WandB 비활성화')
    parser.add_argument('--wandb_project', type=str, default='multi_uav_mission', help='WandB 프로젝트명')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB 엔티티(사용자명 또는 팀명)')
    parser.add_argument('--wandb_tags', type=str, default='', help='WandB 태그 (쉼표로 구분)')
    parser.add_argument('--wandb_notes', type=str, default=None, help='WandB 실행 설명')
    
    # 진행도 표시 관련 설정
    parser.add_argument('--no_tqdm', action='store_true', help='tqdm 진행바 비활성화')
    parser.add_argument('--live_plot', action='store_true', help='실시간 학습 그래프 활성화')
    
    # 저장 및 로깅 관련 설정
    parser.add_argument('--log_interval', type=int, default=10, help='로깅 간격 (스텝)')
    parser.add_argument('--save_interval', type=int, default=10, help='저장 간격 (에포크)')
    parser.add_argument('--verbose', action='store_true', help='상세 로그 출력')
    
    args = parser.parse_args()
    
    # 장치 설정
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"장치: {device}")
    
    # 랜덤 시드 설정
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # 결과 디렉토리 생성
    os.makedirs(args.results_dir, exist_ok=True)
    
    # WandB 설정
    wandb_config = WandBConfig(
        project=args.wandb_project,
        entity=args.wandb_entity,
        tags=args.wandb_tags.split(',') if args.wandb_tags else [],
        notes=args.wandb_notes,
        mode="disabled" if args.no_wandb else "online"
    )
    
    # 모드별 실행
    try:
        if args.mode == 'train':
            run_training(args, device, wandb_config)
        elif args.mode == 'eval':
            run_evaluation(args, device, wandb_config)
        elif args.mode == 'demo':
            run_demo(args, device)
        elif args.mode == 'batch_eval':
            run_batch_evaluation(args, device, wandb_config)
        elif args.mode == 'interactive':
            run_interactive(args, device)
        else:
            print(f"알 수 없는 모드: {args.mode}")
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
        # WandB 종료 (실행 중인 경우)
        if wandb.run:
            wandb.finish()
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        # WandB 종료 (실행 중인 경우)
        if wandb.run:
            wandb.finish()

def run_training(args, device, wandb_config):
    """모델 학습 실행 - 진행도 모니터링 강화"""
    print("\n" + "="*50)
    print("학습 모드 시작")
    print("="*50)
    
    # 시작 시간
    start_time = time.time()
    
    # 데이터 생성
    print("미션 데이터 생성 중...")
    data = MissionData(
        args.num_missions, args.num_uavs, 
        seed=args.seed, device=device
    )
    
    # 데이터 시각화
    viz_path = os.path.join(args.results_dir, "mission_map.png")
    data.visualize(viz_path)
    print(f"미션 맵 시각화를 '{viz_path}'에 저장했습니다.")
    
    # 검증 데이터 생성
    val_data = data.generate_validation_data(args.seed + 1000)
    
    # 설정 객체 생성
    config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        temperature=args.temperature,
        use_curriculum=not args.no_curriculum,
        curriculum_min_missions=args.min_missions,
        hidden_dim=args.hidden_dim,
        num_gat_layers=args.num_gat_layers,
        gat_heads=args.gat_heads,
        transformer_num_layers=args.transformer_layers,
        dropout=args.dropout,
        
        # 저장 및 로깅 설정
        checkpoint_interval=args.save_interval,
        log_interval=args.log_interval,
        verbose=args.verbose,
        live_plot=args.live_plot,
        
        # 진행도 모니터링 설정
        use_wandb=not args.no_wandb,
        wandb_config=wandb_config,
        use_tqdm=not args.no_tqdm
    )
    
    # 환경 생성
    print("학습 환경 초기화 중...")
    train_env = MissionEnvironment(
        data.missions, data.uavs_start, data.uavs_end, 
        data.uavs_speeds, data.uav_types,
        data.risk_centers, data.risk_radii, 
        data.zone_centers, data.zone_radii,
        device, seed=args.seed, 
        curriculum_epoch=0, total_epochs=args.num_epochs,
        min_missions=args.min_missions,
        adaptive_curriculum=args.adaptive_curriculum
    )
    
    # 검증 환경 생성
    val_env = MissionEnvironment(
        val_data.missions, val_data.uavs_start, val_data.uavs_end, 
        val_data.uavs_speeds, val_data.uav_types,
        val_data.risk_centers, val_data.risk_radii, 
        val_data.zone_centers, val_data.zone_radii,
        device, seed=args.seed + 1, 
        curriculum_epoch=0, total_epochs=args.num_epochs,
        min_missions=args.min_missions,
        adaptive_curriculum=args.adaptive_curriculum
    )
    
    # WandB 초기화
    if not args.no_wandb:
        wandb_run = wandb_config.initialize()
        
        # 환경 정보 로깅
        env_config = {
            "environment": {
                "num_uavs": train_env.num_uavs,
                "num_missions": train_env.num_missions,
                "max_missions": train_env.max_missions,
                "risk_areas": train_env.risk_centers.shape[0],
                "no_entry_zones": train_env.zone_centers.shape[0]
            }
        }
        wandb.config.update(env_config)
        
        # 시작 맵 시각화 로깅
        wandb.log({"mission_map": wandb.Image(viz_path, caption="Mission Map")})
    
    # 모델 생성
    print("정책 네트워크 초기화 중...")
    policy_net = TransformerActorCriticNetwork(
        args.num_missions, args.num_uavs, 
        hidden_dim=args.hidden_dim,
        num_gat_layers=args.num_gat_layers,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
        transformer_heads=args.gat_heads,
        transformer_layers=args.transformer_layers
    ).to(device)
    
    # 파라미터 수 출력
    total_params = sum(p.numel() for p in policy_net.parameters())
    trainable_params = sum(p.numel() for p in policy_net.parameters() if p.requires_grad)
    print(f"모델 파라미터: 총 {total_params:,}개, 학습 가능 {trainable_params:,}개")
    
    # WandB에 모델 구조 로깅
    if not args.no_wandb:
        wandb.config.update({
            "model_total_params": total_params,
            "model_trainable_params": trainable_params,
            "model_architecture": {
                "hidden_dim": args.hidden_dim,
                "num_gat_layers": args.num_gat_layers,
                "gat_heads": args.gat_heads,
                "transformer_layers": args.transformer_layers,
                "dropout": args.dropout
            }
        })
        
        # 모델 그래프 감시
        wandb.watch(policy_net, log="all", log_freq=100)
    
    # 옵티마이저 생성
    optimizer_actor = create_optimizer(
        policy_net.actor, args.lr_actor, optimizer_type='adamw'
    )
    optimizer_critic = create_optimizer(
        policy_net.critic, args.lr_critic, optimizer_type='adamw'
    )
    
    # 학습률 스케줄러
    scheduler_actor = create_lr_scheduler(
        optimizer_actor, args.num_epochs, 
        warmup_steps=config.warmup_steps, scheduler_type='cosine'
    )
    scheduler_critic = create_lr_scheduler(
        optimizer_critic, args.num_epochs, 
        warmup_steps=config.warmup_steps, scheduler_type='cosine'
    )
    
    # 엣지 인덱스 캐시
    edge_indices_cache = precompute_edge_indices(args.num_missions, device)
    
    # 학습 설정 요약 출력
    print("\n----- 학습 설정 요약 -----")
    print(f"UAV 수: {args.num_uavs}")
    print(f"미션 수: {args.num_missions}")
    print(f"에포크 수: {args.num_epochs}")
    print(f"배치 크기: {args.batch_size}")
    print(f"학습률: {args.lr_actor} (액터), {args.lr_critic} (크리틱)")
    print(f"커리큘럼 학습: {'사용 안 함' if args.no_curriculum else '사용'}")
    if not args.no_curriculum:
        print(f"최소 미션 수: {args.min_missions}")
        print(f"적응형 커리큘럼: {'사용' if args.adaptive_curriculum else '사용 안 함'}")
    print(f"히든 차원: {args.hidden_dim}")
    print(f"GAT 레이어: {args.num_gat_layers}")
    print(f"GAT 헤드: {args.gat_heads}")
    print(f"트랜스포머 레이어: {args.transformer_layers}")
    print(f"드롭아웃: {args.dropout}")
    print(f"장치: {device}")
    print(f"진행도 표시: {'tqdm 사용 안 함' if args.no_tqdm else 'tqdm 사용'}")
    print(f"WandB: {'사용 안 함' if args.no_wandb else '사용'}")
    print(f"초기 랜덤 시드: {args.seed}")
    print(f"결과 디렉토리: {args.results_dir}")
    if args.checkpoint_path:
        print(f"체크포인트 로드: {args.checkpoint_path}")
    print("-"*30)
    
    # 모델 학습
    print("\n학습 시작...")
    run_name = f"uav_{args.num_uavs}_mission_{args.num_missions}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stats = train_model(
        train_env, val_env, policy_net, 
        optimizer_actor, optimizer_critic, 
        scheduler_actor, scheduler_critic,
        device, edge_indices_cache, config, 
        args.checkpoint_path, args.results_dir,
        run_name=run_name
    )
    
    # 학습 결과 시각화
    visualization_path = os.path.join(args.results_dir, f"{run_name}_training_results.png")
    visualize_training_results(stats, visualization_path)
    print(f"학습 결과 시각화를 '{visualization_path}'에 저장했습니다.")
    
    # 학습 시간
    training_time = time.time() - start_time
    print(f"\n총 학습 시간: {training_time:.1f}초 ({training_time/3600:.2f}시간)")
    
    # 평가 수행
    print("\n학습된 모델 평가 중...")
    best_model_path = os.path.join(args.results_dir, "checkpoints", "best_model.pth")
    if os.path.exists(best_model_path):
        # 베스트 모델 로드
        checkpoint = torch.load(best_model_path, map_location=device)
        policy_net.load_state_dict(checkpoint['model_state_dict'])
        
        # 평가 실행
        eval_metrics = evaluate_model(
            policy_net, val_env, edge_indices_cache, device, config,
            num_episodes=args.eval_episodes,
            render=args.render,
            render_path=os.path.join(args.results_dir, "visualizations"),
            use_wandb=not args.no_wandb,
            wandb_prefix="final_eval"
        )
        
        # 평가 지표 시각화
        evaluation_viz_path = os.path.join(args.results_dir, f"{run_name}_evaluation_metrics.png")
        visualize_evaluation_metrics(eval_metrics, evaluation_viz_path)
        print(f"평가 지표 시각화를 '{evaluation_viz_path}'에 저장했습니다.")
    
    print(f"\n학습 완료. 모든 결과가 '{args.results_dir}'에 저장되었습니다.")
    
    # WandB 종료
    if not args.no_wandb and wandb.run:
        wandb.finish()

def run_evaluation(args, device, wandb_config):
    """모델 평가 실행 - 진행도 표시 강화"""
    print("\n" + "="*50)
    print("평가 모드 시작")
    print("="*50)
    
    # 체크포인트 확인
    if args.checkpoint_path is None:
        print("오류: 평가 모드에는 --checkpoint_path가 필요합니다.")
        return
    
    if not os.path.exists(args.checkpoint_path):
        print(f"오류: 체크포인트 '{args.checkpoint_path}'가 존재하지 않습니다.")
        return
    
    # 시작 시간
    start_time = time.time()
    
    # 체크포인트 로드
    print(f"체크포인트 로드 중: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # 설정 복원 (체크포인트에 저장된 경우)
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("체크포인트에서 설정을 복원했습니다.")
        
        # tqdm 및 wandb 설정 업데이트
        config.use_tqdm = not args.no_tqdm
        config.use_wandb = not args.no_wandb
        config.wandb_config = wandb_config
    else:
        config = TrainingConfig(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            temperature=args.temperature,
            use_curriculum=not args.no_curriculum,
            curriculum_min_missions=args.min_missions,
            hidden_dim=args.hidden_dim,
            num_gat_layers=args.num_gat_layers,
            gat_heads=args.gat_heads,
            transformer_num_layers=args.transformer_layers,
            dropout=args.dropout,
            use_tqdm=not args.no_tqdm,
            use_wandb=not args.no_wandb,
            wandb_config=wandb_config
        )
        print("기본 설정을 사용합니다.")
    
    # 데이터 생성
    print("평가용 미션 데이터 생성 중...")
    data = MissionData(
        args.num_missions, args.num_uavs, 
        seed=args.seed, device=device
    )
    
    # 데이터 시각화
    viz_path = os.path.join(args.results_dir, "eval_mission_map.png")
    data.visualize(viz_path)
    print(f"미션 맵 시각화를 '{viz_path}'에 저장했습니다.")
    
    # WandB 초기화
    if not args.no_wandb:
        wandb_config.job_type = "evaluation"
        wandb_config.tags.append("evaluation")
        wandb_run = wandb_config.initialize()
        
        # 체크포인트 정보 로깅
        wandb.config.update({
            "checkpoint_path": args.checkpoint_path,
            "checkpoint_epoch": checkpoint.get('epoch', 0),
            "num_uavs": args.num_uavs,
            "num_missions": args.num_missions,
            "eval_episodes": args.eval_episodes
        })
        
        # 미션 맵 로깅
        wandb.log({"eval_mission_map": wandb.Image(viz_path, caption="Evaluation Mission Map")})
    
    # 환경 생성
    print("평가 환경 초기화 중...")
    eval_env = MissionEnvironment(
        data.missions, data.uavs_start, data.uavs_end, 
        data.uavs_speeds, data.uav_types,
        data.risk_centers, data.risk_radii, 
        data.zone_centers, data.zone_radii,
        device, seed=args.seed
    )
    
    # 모델 생성
    print("정책 네트워크 초기화 중...")
    policy_net = TransformerActorCriticNetwork(
        args.num_missions, args.num_uavs, 
        hidden_dim=args.hidden_dim,
        num_gat_layers=args.num_gat_layers,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
        transformer_heads=args.gat_heads,
        transformer_layers=args.transformer_layers
    ).to(device)
    
    # 모델 가중치 로드
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    print("모델 가중치를 체크포인트에서 로드했습니다.")
    
    # 엣지 인덱스 캐시
    edge_indices_cache = precompute_edge_indices(args.num_missions, device)
    
    # 결과 디렉토리 생성
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 시각화 경로 설정
    if args.render and args.render_path is None:
        args.render_path = os.path.join(args.results_dir, "eval_visualizations")
    
    # 평가 실행
    print(f"\n{args.eval_episodes}개 에피소드 평가 시작...")
    eval_metrics = evaluate_model(
        policy_net, eval_env, edge_indices_cache, device, config,
        num_episodes=args.eval_episodes,
        render=args.render,
        render_path=args.render_path,
        deterministic=True,
        use_wandb=not args.no_wandb,
        wandb_prefix="eval"
    )
    
    # 평가 지표 시각화
    evaluation_viz_path = os.path.join(args.results_dir, "evaluation_metrics.png")
    visualize_evaluation_metrics(eval_metrics, evaluation_viz_path)
    print(f"평가 지표 시각화를 '{evaluation_viz_path}'에 저장했습니다.")
    
    # 평가 시간
    eval_time = time.time() - start_time
    print(f"\n총 평가 시간: {eval_time:.1f}초")
    
    # WandB 종료
    if not args.no_wandb and wandb.run:
        wandb.finish()
    
    print(f"\n평가 완료. 결과가 '{args.results_dir}'에 저장되었습니다.")

def run_demo(args, device):
    """모델 데모 실행 (대화형 시연)"""
    print("\n" + "="*50)
    print("데모 모드 시작")
    print("="*50)
    
    # 체크포인트 확인
    if args.checkpoint_path is None:
        print("오류: 데모 모드에는 --checkpoint_path가 필요합니다.")
        return
    
    if not os.path.exists(args.checkpoint_path):
        print(f"오류: 체크포인트 '{args.checkpoint_path}'가 존재하지 않습니다.")
        return
    
    # 체크포인트 로드
    print(f"체크포인트 로드 중: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # 데이터 생성
    print("데모용 미션 데이터 생성 중...")
    data = MissionData(
        args.num_missions, args.num_uavs, 
        seed=args.seed, device=device
    )
    
    # 데이터 시각화
    print("\n미션 맵 시각화...")
    data.visualize()
    
    # 환경 생성
    print("데모 환경 초기화 중...")
    demo_env = MissionEnvironment(
        data.missions, data.uavs_start, data.uavs_end, 
        data.uavs_speeds, data.uav_types,
        data.risk_centers, data.risk_radii, 
        data.zone_centers, data.zone_radii,
        device, seed=args.seed
    )
    
    # 모델 생성
    print("정책 네트워크 초기화 중...")
    policy_net = TransformerActorCriticNetwork(
        args.num_missions, args.num_uavs, 
        hidden_dim=args.hidden_dim,
        num_gat_layers=args.num_gat_layers,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
        transformer_heads=args.gat_heads,
        transformer_layers=args.transformer_layers
    ).to(device)
    
    # 모델 가중치 로드
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    print("모델 가중치를 체크포인트에서 로드했습니다.")
    
    # 엣지 인덱스 캐시
    edge_indices_cache = precompute_edge_indices(args.num_missions, device)
    
    # 결과 디렉토리 생성
    os.makedirs(args.results_dir, exist_ok=True)
    demo_path = os.path.join(args.results_dir, "demo")
    os.makedirs(demo_path, exist_ok=True)
    
    # 대화형 데모 시작
    print("\n" + "="*50)
    print("대화형 데모 시작")
    print("="*50)
    print("각 단계에서 Enter를 눌러 진행하고, 'q'를 눌러 종료할 수 있습니다.")
    print("초기 미션 맵을 확인한 후 Enter를 눌러 시작하세요...")
    
    input()
    
    # 대화형 실행
    interactive_evaluation(
        policy_net, demo_env, edge_indices_cache, device,
        config=None, render_path=demo_path, step_by_step=True
    )
    
    print(f"\n데모 완료. 결과가 '{demo_path}'에 저장되었습니다.")

def run_batch_evaluation(args, device, wandb_config):
    """배치 평가 실행 - 여러 환경에서 성능 평가"""
    print("\n" + "="*50)
    print("배치 평가 모드 시작")
    print("="*50)
    
    # 체크포인트 확인
    if args.checkpoint_path is None:
        print("오류: 배치 평가 모드에는 --checkpoint_path가 필요합니다.")
        return
    
    if not os.path.exists(args.checkpoint_path):
        print(f"오류: 체크포인트 '{args.checkpoint_path}'가 존재하지 않습니다.")
        return
    
    # 시작 시간
    start_time = time.time()
    
    # 체크포인트 로드
    print(f"체크포인트 로드 중: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # WandB 초기화
    if not args.no_wandb:
        wandb_config.job_type = "batch_evaluation"
        wandb_config.tags.append("batch_evaluation")
        wandb_run = wandb_config.initialize()
        
        # 배치 평가 설정 로깅
        wandb.config.update({
            "checkpoint_path": args.checkpoint_path,
            "checkpoint_epoch": checkpoint.get('epoch', 0),
            "num_uavs": args.num_uavs,
            "num_missions": args.num_missions,
            "num_envs": args.num_envs,
            "episodes_per_env": args.episodes_per_env
        })
    
    # 모델 클래스 및 인자
    model_args = {
        "max_missions": args.num_missions,
        "max_uavs": args.num_uavs,
        "hidden_dim": args.hidden_dim,
        "num_gat_layers": args.num_gat_layers,
        "gat_heads": args.gat_heads,
        "dropout": args.dropout,
        "transformer_heads": args.gat_heads,
        "transformer_layers": args.transformer_layers
    }
    
    # 환경 생성 함수
    def create_env():
        data = MissionData(
            args.num_missions, args.num_uavs, 
            seed=random.randint(0, 10000),  # 랜덤 시드
            device=device
        )
        
        return MissionEnvironment(
            data.missions, data.uavs_start, data.uavs_end, 
            data.uavs_speeds, data.uav_types,
            data.risk_centers, data.risk_radii, 
            data.zone_centers, data.zone_radii,
            device, seed=random.randint(0, 10000)  # 랜덤 시드
        )
    
    # 엣지 인덱스 캐시
    edge_indices_cache = precompute_edge_indices(args.num_missions, device)
    
    # 결과 디렉토리
    batch_results_dir = os.path.join(args.results_dir, "batch_evaluation")
    os.makedirs(batch_results_dir, exist_ok=True)
    
    # 배치 평가 실행
    print(f"\n{args.num_envs}개 환경, 각 {args.episodes_per_env}개 에피소드로 배치 평가 시작...")
    batch_metrics = batch_evaluation(
        args.checkpoint_path,
        create_env,
        edge_indices_cache,
        device,
        TransformerActorCriticNetwork,
        model_args,
        num_envs=args.num_envs,
        num_episodes_per_env=args.episodes_per_env,
        results_dir=batch_results_dir,
        use_wandb=not args.no_wandb,
        use_tqdm=not args.no_tqdm
    )
    
    # 평가 시간
    eval_time = time.time() - start_time
    print(f"\n총 배치 평가 시간: {eval_time:.1f}초 ({eval_time/60:.2f}분)")
    
    # WandB 종료
    if not args.no_wandb and wandb.run:
        wandb.finish()
    
    print(f"\n배치 평가 완료. 결과가 '{batch_results_dir}'에 저장되었습니다.")

def run_interactive(args, device):
    """대화형 평가 실행 - 모델 결정을 단계별로 시각화"""
    print("\n" + "="*50)
    print("대화형 평가 모드 시작")
    print("="*50)
    
    # 체크포인트 확인
    if args.checkpoint_path is None:
        print("오류: 대화형 평가 모드에는 --checkpoint_path가 필요합니다.")
        return
    
    if not os.path.exists(args.checkpoint_path):
        print(f"오류: 체크포인트 '{args.checkpoint_path}'가 존재하지 않습니다.")
        return
    
    # 체크포인트 로드
    print(f"체크포인트 로드 중: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # 데이터 생성
    print("평가용 미션 데이터 생성 중...")
    data = MissionData(
        args.num_missions, args.num_uavs, 
        seed=args.seed, device=device
    )
    
    # 데이터 시각화
    viz_path = os.path.join(args.results_dir, "interactive_mission_map.png")
    data.visualize(viz_path)
    print(f"미션 맵 시각화를 '{viz_path}'에 저장했습니다.")
    
    # 환경 생성
    print("평가 환경 초기화 중...")
    eval_env = MissionEnvironment(
        data.missions, data.uavs_start, data.uavs_end, 
        data.uavs_speeds, data.uav_types,
        data.risk_centers, data.risk_radii, 
        data.zone_centers, data.zone_radii,
        device, seed=args.seed
    )
    
    # 모델 생성
    print("정책 네트워크 초기화 중...")
    policy_net = TransformerActorCriticNetwork(
        args.num_missions, args.num_uavs, 
        hidden_dim=args.hidden_dim,
        num_gat_layers=args.num_gat_layers,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
        transformer_heads=args.gat_heads,
        transformer_layers=args.transformer_layers
    ).to(device)
    
    # 모델 가중치 로드
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    print("모델 가중치를 체크포인트에서 로드했습니다.")
    
    # 엣지 인덱스 캐시
    edge_indices_cache = precompute_edge_indices(args.num_missions, device)
    
    # 결과 디렉토리 생성
    os.makedirs(args.results_dir, exist_ok=True)
    interactive_path = os.path.join(args.results_dir, "interactive")
    os.makedirs(interactive_path, exist_ok=True)
    
    # 대화형 평가 시작
    print("\n" + "="*50)
    print("대화형 평가 시작")
    print("="*50)
    print("초기 미션 맵을 확인한 후 Enter를 눌러 시작하세요...")
    
    input()
    
    # 대화형 평가 실행
    interactive_evaluation(
        policy_net, eval_env, edge_indices_cache, device,
        config=None, render_path=interactive_path, step_by_step=True
    )
    
    print(f"\n대화형 평가 완료. 결과가 '{interactive_path}'에 저장되었습니다.")

if __name__ == "__main__":
    # 예외 처리를 포함한 메인 함수 실행
    try:
        main()
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 중단되었습니다.")
        # WandB 종료 (실행 중인 경우)
        if wandb.run:
            wandb.finish()
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        # WandB 종료 (실행 중인 경우)
        if wandb.run:
            wandb.finish()