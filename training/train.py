# training/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
import wandb
from datetime import datetime

from utils.calculations import calculate_cost_matrix, calculate_arrival_times
from utils.masks import create_action_mask
from utils.visualization import visualize_results
from utils.rewards import compute_reward_max_time, compute_reward_total_time, compute_reward_mixed
from environment.mission_environment import MissionEnvironment

def choose_action(action_probs, epsilon=0.1, uav_order=None, global_action_mask=None):
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

def compute_uav_order(env):
    """
    예상 도착 시간을 기반으로 UAV 선택 순서를 결정합니다.
    """
    expected_arrival_times = []
    for i in range(env.num_uavs):
        if env.ready_for_next_action[i]:
            expected_arrival_times.append(0.0)
        else:
            if env.remaining_distances[i] == float('inf'):
                expected_arrival_times.append(float('inf'))
            else:
                expected_time = env.remaining_distances[i].item() / (env.speeds[i].item() + 1e-5)
                expected_arrival_times.append(expected_time)
    
    uav_indices = list(range(env.num_uavs))
    uav_order = sorted(uav_indices, key=lambda i: (expected_arrival_times[i], -env.speeds[i].item()))
    return uav_order

def train_model(env, val_env, policy_net, optimizer_actor, optimizer_critic, scheduler_actor, scheduler_critic,
               num_epochs, batch_size, device, edge_index, batch, epsilon_decay, gamma, 
               reward_type='total', alpha=0.5, beta=0.5,
               start_epoch=1, checkpoint_path=None, results_path=None, checkpoints_path=None, patience=10):
    """
    액터-크리틱 정책 네트워크를 학습합니다.
    """
    # WandB 초기화
    wandb.init(project="multi_uav_mission", name="train_without_2opt_test", config={
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate_actor": optimizer_actor.param_groups[0]['lr'],
        "learning_rate_critic": optimizer_critic.param_groups[0]['lr'],
        "weight_decay_actor": optimizer_actor.param_groups[0]['weight_decay'],
        "weight_decay_critic": optimizer_critic.param_groups[0]['weight_decay'],
        "epsilon_decay": epsilon_decay,
        "gamma": gamma,
        "patience": patience,
        "gnn_dropout": policy_net.gnn_encoder.gnn_output[1].p,
        "actor_dropout": policy_net.actor_fc[2].p,
        "critic_dropout": policy_net.critic_fc[2].p,
        "num_missions": env.num_missions,
        "num_uavs": env.num_uavs,
        "reward_type": reward_type,
        "alpha": alpha,
        "beta": beta
    })
    
    epsilon = 1.0
    epsilon_min = 0.1

    # 체크포인트 로드
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 액터와 크리틱의 마지막 레이어를 제외한 상태 사전 로드
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith('module.actor_fc.3') and not k.startswith('module.critic_fc.3'):  # DataParallel 사용 시 'module.' 추가
                new_state_dict[k.replace('module.', '')] = v  # DataParallel 사용 시 'module.' 제거
        
        # 모델 상태 업데이트 (strict=False로 일치하지 않는 키는 무시)
        policy_net.load_state_dict(new_state_dict, strict=False)

        # 마지막 레이어 초기화
        nn.init.xavier_uniform_(policy_net.actor_fc[3].weight)
        nn.init.zeros_(policy_net.actor_fc[3].bias)
        nn.init.xavier_uniform_(policy_net.critic_fc[3].weight)
        nn.init.zeros_(policy_net.critic_fc[3].bias)

        # 옵티마이저 상태도 로드
        optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        epsilon = checkpoint['epsilon']
        print(f"체크포인트 '{checkpoint_path}'가 로드되었습니다. {start_epoch} 에폭부터 시작합니다.")

    total_episodes = num_epochs * batch_size
    episode = (start_epoch - 1) * batch_size

    best_validation_reward = -float('inf')
    epochs_no_improve = 0

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            # tqdm 인스턴스 생성
            epoch_pbar = tqdm(range(batch_size), desc=f"에폭 {epoch}/{num_epochs}", leave=False)
            for batch_idx in epoch_pbar:
                state = env.reset()
                done = False
                log_probs = []
                values = []
                rewards = []
                entropy_list = []  # 엔트로피를 저장할 리스트
                travel_times = []  # 이동 시간 기록

                while not done:
                    positions = state['positions']
                    uavs_info = positions.to(device)
                    action_mask = create_action_mask(state)
                    
                    # 비용 행렬과 도착 시간 계산
                    cost_matrix = calculate_cost_matrix(positions, env.missions, env.speeds)
                    arrival_times = calculate_arrival_times(positions, env.missions, env.speeds)

                    # 정책 네트워크 순전파
                    action_probs, state_values = policy_net(
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
                        state_values = state_values.mean(dim=0)  # 동일하게 병합

                    # UAV 선택 순서 결정
                    uav_order = compute_uav_order(env)
                    
                    # 액션 선택
                    actions = choose_action(action_probs, epsilon, uav_order, global_action_mask=action_mask)

                    # 각 UAV의 액션에 대한 log_prob과 state_value 수집
                    for i, action in enumerate(actions):
                        # 선택된 액션의 확률을 가져옵니다.
                        prob = action_probs[i, action]
                        log_prob = torch.log(prob + 1e-10).squeeze()  # 스칼라로 만듦
                        log_probs.append(log_prob)
                        values.append(state_values[i].squeeze())  # 스칼라로 만듦

                    # 엔트로피 계산 및 저장
                    entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=1).mean()
                    entropy_list.append(entropy)

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

                    rewards.append(reward)
                    state = next_state

                    # 이동 시간 기록
                    travel_times.append(env.cumulative_travel_times.clone())

                # 할인율을 적용한 누적 보상 계산
                returns = []
                R = 0
                for r in reversed(rewards):
                    R = r + gamma * R
                    returns.insert(0, R)
                returns = torch.tensor(returns, device=device)

                # 보상 표준화 (선택 사항)
                if returns.std() != 0:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-5)

                policy_loss = []
                value_loss = []
                for log_prob, value, R in zip(log_probs, values, returns):
                    advantage = R - value
                    policy_loss.append(-log_prob * advantage)
                    value_loss.append(F.mse_loss(value, R.unsqueeze(0)))

                if policy_loss and value_loss:
                    # 정책 손실과 가치 손실의 평균을 취함
                    policy_loss_total = torch.stack(policy_loss).mean()

                    # 엔트로피 보너스 추가 (탐험 유도)
                    entropy_total = torch.stack(entropy_list).mean()
                    policy_loss_total = policy_loss_total - 0.01 * entropy_total

                    value_loss_total = torch.stack(value_loss).mean()
                    loss = policy_loss_total + value_loss_total
                    
                    # 역전파
                    optimizer_actor.zero_grad()
                    optimizer_critic.zero_grad()
                    loss.backward()
                    
                    # 그래디언트 클리핑
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                    
                    # 옵티마이저 스텝
                    optimizer_actor.step()
                    optimizer_critic.step()
                else:
                    # policy_loss와 value_loss가 비어 있을 경우, 손실을 0으로 설정
                    loss = torch.tensor(0.0, device=device)

                # epsilon 업데이트
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay

                # 보상과 이동 시간 로깅
                average_travel_time = torch.stack(travel_times).mean().item() if travel_times else 0.0

                # tqdm 진행 표시줄에 정보 업데이트
                epoch_pbar.set_description(f"에폭 {epoch}/{num_epochs} | 배치 {batch_idx+1}/{batch_size} | 보상 {rewards[-1]:.2f} | 손실 {loss.item():.4f} | Epsilon {epsilon:.4f}")

                # WandB에 로그 기록
                wandb.log({
                    "episode": episode,
                    "epoch": epoch,
                    "batch": batch_idx,
                    "policy_loss": policy_loss_total.item() if policy_loss else 0,
                    "value_loss": value_loss_total.item() if value_loss else 0,
                    "loss": loss.item(),
                    "reward": rewards[-1],
                    "epsilon": epsilon,
                    "entropy": entropy_total.item() if policy_loss and value_loss else 0,
                    "average_travel_time": average_travel_time,  # 평균 이동 시간 추가
                    "uav_travel_times": env.cumulative_travel_times.tolist(),  # UAV별 이동 시간 추가
                    "uav_assignments": [len(path) for path in env.paths]  # UAV별 할당된 미션 수
                })

                # 가시화 및 체크포인트 저장
                if episode % 1000 == 0:
                    visualization_path = os.path.join(results_path, f"mission_paths_episode_{episode}.png")
                    visualize_results(
                        env, 
                        visualization_path,
                        reward=rewards[-1],
                        epsilon=epsilon,
                        policy_loss=policy_loss_total.item() if policy_loss else 0,
                        value_loss=value_loss_total.item() if value_loss else 0,
                        folder_name=results_path
                    )
                    wandb.log({
                        "mission_paths": wandb.Image(visualization_path)
                    })

                episode += 1

            # tqdm 인스턴스 종료
            epoch_pbar.close()

            # 학습률 스케줄러 업데이트
            scheduler_actor.step()
            scheduler_critic.step()

            # 검증
            if epoch % 10 == 0:
                from training.validate import validate_model
                validation_reward = validate_model(val_env, policy_net, device, edge_index, batch, checkpoints_path, results_path, epoch, reward_type, alpha, beta)
                
                # 조기 종료 체크
                if validation_reward > best_validation_reward:
                    best_validation_reward = validation_reward
                    epochs_no_improve = 0
                    # 최적의 모델을 별도로 저장
                    best_model_path = os.path.join(checkpoints_path, f"best_model_epoch_{epoch}.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': policy_net.state_dict(),
                        'optimizer_actor_state_dict': optimizer_actor.state_dict(),
                        'optimizer_critic_state_dict': optimizer_critic.state_dict(),
                        'epsilon': epsilon
                    }, best_model_path)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"조기 종료가 {patience} 에폭 동안 개선되지 않아 트리거되었습니다.")
                        return

    except KeyboardInterrupt:
        print("학습이 중단되었습니다. 체크포인트를 저장합니다...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_actor_state_dict': optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': optimizer_critic.state_dict(),
            'epsilon': epsilon
        }, os.path.join(checkpoints_path, f"interrupted_epoch_{epoch}.pth"))
        print("체크포인트가 저장되었습니다. 안전하게 종료합니다.")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wandb.finish()
