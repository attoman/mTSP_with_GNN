# test.py

import os
import torch
from utils import create_action_mask, calculate_cost_matrix, calculate_arrival_times, choose_action, compute_uav_order, visualize_results
import wandb

def test_model(env, policy_net, device, edge_index, batch, checkpoint_path, results_path):
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy_net.load_state_dict(checkpoint['model_state_dict'])
        print(f"체크포인트 '{checkpoint_path}'가 로드되었습니다. 테스트를 시작합니다.")
    else:
        print("유효한 체크포인트가 제공되지 않았습니다.")
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
        
        # 비용 행렬 계산
        cost_matrix = calculate_cost_matrix(positions, env.missions, env.speeds)
        arrival_times = calculate_arrival_times(positions, env.missions, env.speeds)

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
        
        # UAV 선택 순서 결정
        uav_order = compute_uav_order(env)
        # 액션 선택 (epsilon=0.0으로 설정하여 탐색 없이 정책에 따름)
        actions = choose_action(action_probs, epsilon=0.0, uav_order=uav_order, global_action_mask=action_mask)

        next_state, reward, done = env.step(actions)
        total_reward += reward
        state = next_state

        for i in range(env.num_uavs):
            paths[i] = env.paths[i]
            cumulative_travel_times[i] = env.cumulative_travel_times[i]

    print(f"테스트 완료 - 총 보상: {total_reward:.2f}")
    visualization_path = os.path.join(results_path, "test_results.png")
    visualize_results(
        env, 
        visualization_path,
        reward=total_reward,
        epsilon=None,
        policy_loss=None,
        value_loss=None,
        folder_name='Test Results'
    )
    wandb.log({
        "test_reward": total_reward,
        "test_cumulative_travel_times": cumulative_travel_times.tolist(),
        "test_mission_paths": wandb.Image(visualization_path)
    })
