# validate.py

import os
import torch
from utils import create_action_mask, calculate_cost_matrix, calculate_arrival_times, choose_action, compute_uav_order, visualize_results
import wandb

def validate_model(env, policy_net, device, edge_index, batch, checkpoints_best_path, images_best_path, epoch, best_validation_reward):
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

    # 검증 결과 저장
    if total_reward > best_validation_reward:
        best_validation_reward = total_reward
        # 최적의 모델을 'checkpoints_best' 폴더에 저장
        best_model_path = os.path.join(checkpoints_best_path, f"best_model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_actor_state_dict': None,  # 크리틱과 옵티마이저는 필요 시 저장
            'optimizer_critic_state_dict': None,
            'epsilon': None  # 필요한 경우 추가
        }, best_model_path)
        
        # 최적의 이미지도 'images_best' 폴더에 저장
        visualization_path_best = os.path.join(images_best_path, f"best_mission_paths_epoch_{epoch}.png")
        visualize_results(
            env, 
            visualization_path_best,
            reward=total_reward,
            epsilon=None,
            policy_loss=None,
            value_loss=None,
            folder_name='Best Results'
        )
        wandb.log({
            "best_mission_paths": wandb.Image(visualization_path_best),
            "best_validation_reward": total_reward
        })
        print(f"[Epoch {epoch}] 새로운 최고 검증 보상을 달성했습니다: {total_reward:.2f}")
    else:
        print(f"[Epoch {epoch}] 검증 보상: {total_reward:.2f} (현재 최고: {best_validation_reward:.2f})")

    return best_validation_reward
