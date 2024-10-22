# main.py

import os
import argparse
from datetime import datetime
import torch
import torch.optim as optim
from models import ImprovedActorCriticNetwork
from data import MissionData
from environment import MissionEnvironment
from train import train_model
from test import test_model
from utils import create_edge_index
import wandb


def main():
    """
    메인 함수: 학습 또는 테스트 모드를 선택하여 실행합니다.
    """
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="mTSP Actor-Critic GNN with 2-opt Optimization and Per-Step Rewards")
    parser.add_argument('--gpu', type=int, default=0, help="사용할 GPU 인덱스")
    parser.add_argument('--num_uavs', type=int, default=3, help="UAV의 수")
    parser.add_argument('--num_missions', type=int, default=12, help="미션의 수")
    parser.add_argument('--embedding_dim', type=int, default=64, help="GNN 임베딩 차원")
    parser.add_argument('--hidden_dim', type=int, default=128, help="FC 레이어의 은닉 차원")
    parser.add_argument('--num_layers', type=int, default=4, help="GNN 레이어 수")
    parser.add_argument('--heads', type=int, default=8, help="GNN Transformer 헤드 수")
    parser.add_argument('--num_epochs', type=int, default=3000, help="에폭 수")
    parser.add_argument('--batch_size', type=int, default=100, help="배치 크기")
    parser.add_argument('--epsilon_decay', type=float, default=0.9999, help="Epsilon 감소율")
    parser.add_argument('--gamma', type=float, default=0.99, help="할인율 (gamma)")
    parser.add_argument('--lr_actor', type=float, default=1e-4, help="액터 학습률")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="크리틱 학습률")
    parser.add_argument('--weight_decay_actor', type=float, default=1e-5, help="액터 옵티마이저의 weight decay")
    parser.add_argument('--weight_decay_critic', type=float, default=1e-5, help="크리틱 옵티마이저의 weight decay")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="기존 체크포인트의 경로")
    parser.add_argument('--test_mode', action='store_true', help="테스트 모드 활성화")
    parser.add_argument('--validation_seed', type=int, default=43, help="Validation 데이터셋 시드")
    parser.add_argument('--test_seed', type=int, default=44, help="Test 데이터셋 시드")
    parser.add_argument('--time_weight', type=float, default=1.0, help="이동 시간에 대한 보상 가중치")
    parser.add_argument('--results_dir', type=str, default="./results/", help="결과 저장 디렉토리")
    args = parser.parse_args()

    # 시드 설정
    from utils import set_seed
    set_seed(42)

    # 디바이스 설정
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        print(f"GPU {args.gpu} 사용 중: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("CPU 사용 중")

    # 데이터 생성
    train_data = MissionData(
        num_missions=args.num_missions,
        num_uavs=args.num_uavs,
        seed=42,
        device=device
    )
    val_data = MissionData(
        num_missions=args.num_missions,
        num_uavs=args.num_uavs,
        seed=args.validation_seed,
        device=device
    )
    test_data = MissionData(
        num_missions=args.num_missions,
        num_uavs=args.num_uavs,
        seed=args.test_seed,
        device=device
    )

    # 환경 초기화
    train_env = MissionEnvironment(
        missions=train_data.missions,
        uavs_start=train_data.uavs_start,
        uavs_speeds=train_data.uavs_speeds,
        device=device,
        mode='train',
        seed=42,
        time_weight=args.time_weight
    )
    val_env = MissionEnvironment(
        missions=val_data.missions,
        uavs_start=val_data.uavs_start,
        uavs_speeds=val_data.uavs_speeds,
        device=device,
        mode='val',
        seed=args.validation_seed,
        time_weight=args.time_weight
    )
    test_env = MissionEnvironment(
        missions=test_data.missions,
        uavs_start=test_data.uavs_start,
        uavs_speeds=test_data.uavs_speeds,
        device=device,
        mode='test',
        seed=args.test_seed,
        time_weight=args.time_weight
    )

    # 엣지 인덱스 및 배치 생성
    edge_index = create_edge_index(args.num_missions, args.num_uavs).to(device)
    batch = torch.arange(args.num_uavs).repeat_interleave(args.num_missions).to(device)  # [0,0,...0,1,1,...1,2,2,...2]

    # 네트워크 초기화
    policy_net = ImprovedActorCriticNetwork(
        num_missions=args.num_missions,
        num_uavs=args.num_uavs,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads
    ).to(device)
    
    # 옵티마이저 초기화
    optimizer_actor = optim.Adam(
        policy_net.actor_fc.parameters(),
        lr=args.lr_actor,
        weight_decay=args.weight_decay_actor
    )
    optimizer_critic = optim.Adam(
        policy_net.critic_fc.parameters(),
        lr=args.lr_critic,
        weight_decay=args.weight_decay_critic
    )

    # 현재 스크립트 디렉토리 기준으로 'results' 폴더 경로 정의
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(
        script_dir, 
        'results', 
        f"num_missions_{args.num_missions}", 
        "revision", 
        "images", 
        current_time
    )
    checkpoints_path = os.path.join(
        script_dir, 
        'results', 
        f"num_missions_{args.num_missions}", 
        "revision", 
        "checkpoints", 
        current_time
    )
    
    # 'checkpoints_best'와 'images_best' 경로 정의 및 생성
    checkpoints_best_path = os.path.join(script_dir, 'results', 'checkpoints_best')
    images_best_path = os.path.join(script_dir, 'results', 'images_best')
    os.makedirs(checkpoints_best_path, exist_ok=True)
    os.makedirs(images_best_path, exist_ok=True)

    # 기본 'results'와 'checkpoints' 폴더 생성
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    # 학습 모드 또는 테스트 모드에 따라 실행
    if args.test_mode:
        test_model(
            env=test_env, 
            policy_net=policy_net, 
            device=device, 
            edge_index=edge_index, 
            batch=batch, 
            checkpoint_path=args.checkpoint_path,
            results_path=results_path
        )
    else:
        train_model(
            env=train_env, 
            val_env=val_env, 
            policy_net=policy_net, 
            optimizer_actor=optimizer_actor,
            optimizer_critic=optimizer_critic,
            num_epochs=args.num_epochs, 
            batch_size=args.batch_size, 
            device=device, 
            edge_index=edge_index, 
            batch=batch, 
            epsilon_decay=args.epsilon_decay, 
            gamma=args.gamma,  # 할인율 전달
            checkpoint_path=args.checkpoint_path,
            results_path=results_path,
            checkpoints_path=checkpoints_path,
            checkpoints_best_path=checkpoints_best_path,  # 추가
            images_best_path=images_best_path,          # 추가
            patience=10  # patience 값을 원하는 만큼 조정할 수 있습니다.
        )


if __name__ == "__main__":
    main()
