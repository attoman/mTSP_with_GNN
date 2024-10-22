# training/test.py

import os
import torch
import wandb
import random
import torch.nn.functional as F
from utils.masks import create_action_mask
from utils.calculations import calculate_cost_matrix, calculate_arrival_times
from utils.visualization import visualize_results
from utils.rewards import compute_reward_max_time, compute_reward_total_time, compute_reward_mixed
from environment.mission_environment import MissionEnvironment

def choose_action(action_probs, epsilon=0.0, uav_order=None, global_action_mask=None):
    """
    각 UAV에 대해 액션을 선택합니다. epsilon-탐욕 전략을 사용합니다.
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
        local_action_mask[action] = True  # 다른 UAV가 동일한 액션을 선택하지 않도록 마스크
    
    return actions

def test_model(env, policy_net, device, edge_index, batch, checkpoint_path, results_path, reward_type='total', alpha=0.5, beta=0.5):
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
    """
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
        from training.train import compute_uav_order
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

    print(f"테스트 완료 - 총 보상: {total_reward}")
    visualization_path = os.path.join(results_path, "test_results.png")
    from utils.visualization import visualize_results
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
        "test_reward": total_reward,
        "test_cumulative_travel_times": cumulative_travel_times.tolist(),
        "test_mission_paths": wandb.Image(visualization_path)
    })
