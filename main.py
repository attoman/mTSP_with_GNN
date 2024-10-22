# main.py

import os
import torch
import torch.optim as optim
from datetime import datetime
import argparse

from utils.seed import set_seed
from data.mission_data import MissionData
from environment.mission_environment import MissionEnvironment
from models.actor_critic_network import ImprovedActorCriticNetwork
from training.train import train_model
from training.test import test_model

def main():
    """
    인자를 파싱하고 학습 또는 테스트를 시작하는 메인 함수.
    """
    parser = argparse.ArgumentParser(description="액터-크리틱 GNN을 이용한 다중 UAV 미션 할당 및 최적화")
    parser.add_argument('--gpu', type=str, default='0', help="사용할 GPU 인덱스 (예: '0', '0,1', '0,1,2,3')")
    parser.add_argument('--num_uavs', type=int, default=3, help="UAV의 수")
    parser.add_argument('--num_missions', type=int, default=12, help="미션의 수")
    parser.add_argument('--embedding_dim', type=int, default=64, help="GNN 임베딩 차원")
    parser.add_argument('--hidden_dim', type=int, default=128, help="FC 레이어의 은닉 차원")
    parser.add_argument('--num_layers', type=int, default=4, help="GNN 레이어 수")
    parser.add_argument('--heads', type=int, default=8, help="GNN Transformer 헤드 수")
    parser.add_argument('--num_epochs', type=int, default=6000, help="에폭 수")
    parser.add_argument('--batch_size', type=int, default=500, help="배치 크기")
    parser.add_argument('--epsilon_decay', type=float, default=0.9999, help="Epsilon 감소율")
    parser.add_argument('--gamma', type=float, default=0.999, help="할인율 (gamma)")
    parser.add_argument('--lr_actor', type=float, default=1e-4, help="액터 학습률")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="크리틱 학습률")
    parser.add_argument('--weight_decay_actor', type=float, default=1e-5, help="액터 옵티마이저의 weight decay")
    parser.add_argument('--weight_decay_critic', type=float, default=1e-5, help="크리틱 옵티마이저의 weight decay")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="기존 체크포인트의 경로")
    parser.add_argument('--test_mode', action='store_true', help="테스트 모드 활성화")
    parser.add_argument('--train_seed', type=int, default=2024, help="Train 데이터셋 시드")
    parser.add_argument('--validation_seed', type=int, default=43, help="Validation 데이터셋 시드")
    parser.add_argument('--test_seed', type=int, default=44, help="Test 데이터셋 시드")
    parser.add_argument('--time_weight', type=float, default=2.0, help="보상 시간의 가중치")
    parser.add_argument('--lr_step_size', type=int, default=1000, help="학습률 스케줄러의 step size")
    parser.add_argument('--lr_gamma', type=float, default=0.1, help="학습률 스케줄러의 gamma 값")
    
    # 드롭아웃 비율을 조정할 수 있는 인자 추가
    parser.add_argument('--gnn_dropout', type=float, default=0.3, help="GNN Transformer 인코더의 드롭아웃 비율")
    parser.add_argument('--actor_dropout', type=float, default=0.3, help="액터 네트워크의 드롭아웃 비율")
    parser.add_argument('--critic_dropout', type=float, default=0.3, help="크리틱 네트워크의 드롭아웃 비율")
    
    # 보상 함수 선택 인자 추가
    parser.add_argument('--reward_type', type=str, default='total', choices=['max', 'total', 'mixed'], help="보상 함수 유형: 'max', 'total', 'mixed'")
    parser.add_argument('--alpha', type=float, default=0.5, help="혼합 보상 시 최대 소요 시간 패널티 가중치 (reward_type='mixed'일 때 사용)")
    parser.add_argument('--beta', type=float, default=0.5, help="혼합 보상 시 전체 소요 시간 합 패널티 가중치 (reward_type='mixed'일 때 사용)")
    
    # 결과 디렉토리 추가
    parser.add_argument('--results_dir', type=str, default="./results/", help="결과 저장 디렉토리")
    
    args = parser.parse_args()

    # 장치 설정
    # GPU 인자 처리
    gpu_indices = [int(x) for x in args.gpu.split(',')]
    num_gpus = len(gpu_indices)
    if num_gpus > 8:
        raise ValueError("최대 8개의 GPU만 지원됩니다.")
    for gpu in gpu_indices:
        if gpu < 0 or gpu >= torch.cuda.device_count():
            raise ValueError(f"GPU 인덱스 {gpu}는 사용 불가능합니다. 사용 가능한 GPU 인덱스: 0-{torch.cuda.device_count()-1}")
    
    if num_gpus > 1:
        device = torch.device(f"cuda:{gpu_indices[0]}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_indices))
        print(f"{num_gpus}개의 GPU {gpu_indices}를 사용합니다.")
    elif num_gpus == 1:
        device = torch.device(f"cuda:{gpu_indices[0]}")
        print(f"GPU {gpu_indices[0]}를 사용합니다.")
    else:
        device = torch.device("cpu")
        print("CPU를 사용합니다.")

    # 재현성을 위해 시드 설정
    set_seed(args.train_seed)

    # 데이터 생성
    train_data = MissionData(num_missions=args.num_missions, num_uavs=args.num_uavs, seed=args.train_seed, device=device)
    val_data = MissionData(num_missions=args.num_missions, num_uavs=args.num_uavs, seed=args.validation_seed, device=device)
    test_data = MissionData(num_missions=args.num_missions, num_uavs=args.num_uavs, seed=args.test_seed, device=device)

    # 환경 초기화
    train_env = MissionEnvironment(train_data.missions, train_data.uavs_start, train_data.uavs_speeds, device, mode='train', seed=args.train_seed, time_weight=args.time_weight)
    val_env = MissionEnvironment(val_data.missions, val_data.uavs_start, val_data.uavs_speeds, device, mode='val', seed=args.validation_seed, time_weight=args.time_weight)
    test_env = MissionEnvironment(test_data.missions, test_data.uavs_start, test_data.uavs_speeds, device, mode='test', seed=args.test_seed, time_weight=args.time_weight)

    # edge_index와 batch 생성
    from utils.masks import create_edge_index
    edge_index = create_edge_index(args.num_missions, args.num_uavs).to(device)
    batch = torch.arange(args.num_uavs).repeat_interleave(args.num_missions).to(device)

    # 정책 네트워크 초기화
    policy_net = ImprovedActorCriticNetwork(
        num_missions=args.num_missions,
        num_uavs=args.num_uavs,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        gnn_dropout=args.gnn_dropout,
        actor_dropout=args.actor_dropout,
        critic_dropout=args.critic_dropout
    ).to(device)
    
    # DataParallel 사용 설정 (선택 사항)
    if num_gpus > 1:
        policy_net = nn.DataParallel(policy_net)

    # 옵티마이저 초기화
    optimizer_actor = optim.Adam(policy_net.actor_fc.parameters(), lr=args.lr_actor, weight_decay=args.weight_decay_actor)
    optimizer_critic = optim.Adam(policy_net.critic_fc.parameters(), lr=args.lr_critic, weight_decay=args.weight_decay_critic)

    # 학습률 스케줄러 초기화
    scheduler_actor = optim.lr_scheduler.StepLR(optimizer_actor, step_size=args.lr_step_size, gamma=args.lr_gamma)
    scheduler_critic = optim.lr_scheduler.StepLR(optimizer_critic, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # 결과 및 체크포인트 디렉토리 생성
    num_missions_folder = f"num_missions_{args.num_missions}"
    revision_folder = "revision"
    
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"현재 시간: {current_time}")
    results_path = os.path.join(args.results_dir, num_missions_folder, revision_folder, "images", current_time)
    checkpoints_path = os.path.join(args.results_dir, num_missions_folder, revision_folder, "checkpoints", current_time)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    # 학습 또는 테스트 모드 실행
    if args.test_mode:
        test_model(env=test_env, 
                   policy_net=policy_net, 
                   device=device, 
                   edge_index=edge_index, 
                   batch=batch, 
                   checkpoint_path=args.checkpoint_path,
                   results_path=results_path,
                   reward_type=args.reward_type,
                   alpha=args.alpha,
                   beta=args.beta)
    else:
        train_model(env=train_env, 
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
                    epsilon_decay=args.epsilon_decay, 
                    gamma=args.gamma,
                    reward_type=args.reward_type,
                    alpha=args.alpha,
                    beta=args.beta,
                    checkpoint_path=args.checkpoint_path,
                    results_path=results_path,
                    checkpoints_path=checkpoints_path,
                    patience=10)

if __name__ == "__main__":
    main()
