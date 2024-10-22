# training/validate.py

import os
import torch
import wandb
import torch.nn as nn
from utils.masks import create_action_mask
from utils.calculations import calculate_cost_matrix, calculate_arrival_times
from utils.visualization import visualize_results
from utils.rewards import compute_reward_max_time, compute_reward_total_time, compute_reward_mixed

def validate_model(env, policy_net, device, edge_index, batch, checkpoints_path, results_path, epoch, reward_type, alpha, beta):
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
        
    Returns:
        float: 총 검증 보상.
    """
    state = env.reset()
    done = False
    total_reward = 0
    cumulative_travel_times = torch.zeros(env.num_uavs, device=device)
    paths = [[] for _ in range(env.num_uavs)]

    while not done:
        positions = state['positions']
        uavs_info = positions.to(device)
        action_mask = create_action_mask(state)
        
        # 비용 행렬과 도착 시간 계산
        cost_matrix = calculate_cost_matrix(positions, env.missions, env.speeds)
        arrival_times = calculate_arrival_times(positions, env.missions, env.speeds)

        # 정책 네트워크 순전파
        action_probs, _ = policy_net(
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

        # UAV 선택 순서 결정
        from training.train import choose_action, compute_uav_order
        uav_order = compute_uav_order(env)
        # 탐험 없이 액터 정책에 따라 액션 선택 (epsilon=0.0)
        actions = choose_action(action_probs, epsilon=0.0, uav_order=uav_order, global_action_mask=action_mask)

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
        folder_name=results_path
    )
    wandb.log({
        "validation_reward": total_reward,
        "validation_cumulative_travel_times": cumulative_travel_times.tolist(),
        "validation_mission_paths": wandb.Image(visualization_path),
        "epoch": epoch
    })

    return total_reward
