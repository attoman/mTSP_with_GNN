# train.py

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import wandb
from models import ImprovedActorCriticNetwork
from utils import (
    choose_action,
    compute_uav_order,
    visualize_results,
    calculate_cost_matrix,
    calculate_arrival_times,
    create_action_mask
)
from validate import validate_model


def train_model(
    env,
    val_env,
    policy_net,
    optimizer_actor,
    optimizer_critic,
    num_epochs,
    batch_size,
    device,
    edge_index,
    batch,
    epsilon_decay,
    gamma,
    start_epoch=1,
    checkpoint_path=None,
    results_path=None,
    checkpoints_path=None,
    checkpoints_best_path=None,
    images_best_path=None,
    patience=10
):
    """
    모델을 학습시키는 함수입니다.

    Args:
        env: 학습 환경.
        val_env: 검증 환경.
        policy_net: 학습할 정책 신경망.
        optimizer_actor: 액터 네트워크의 옵티마이저.
        optimizer_critic: 크리틱 네트워크의 옵티마이저.
        num_epochs: 전체 학습 에폭 수.
        batch_size: 에폭 당 배치 수.
        device: 학습에 사용할 디바이스 (CPU 또는 GPU).
        edge_index: 그래프의 엣지 인덱스.
        batch: 배치 정보.
        epsilon_decay: 탐험율 감소율.
        gamma: 할인율.
        start_epoch: 학습을 시작할 에폭 번호.
        checkpoint_path: 로드할 체크포인트의 경로.
        results_path: 시각화 결과를 저장할 경로.
        checkpoints_path: 체크포인트를 저장할 경로.
        checkpoints_best_path: 최고 성능 모델을 저장할 경로.
        images_best_path: 최고 성능 시각화를 저장할 경로.
        patience: 조기 종료를 위한 인내 에폭 수.
    """
    # WandB 초기화
    wandb.init(project="multi_uav_mission", name="train_2opt_sumtime_rev")
    wandb.config.update({
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate_actor": optimizer_actor.param_groups[0]['lr'],
        "learning_rate_critic": optimizer_critic.param_groups[0]['lr'],
        "weight_decay_actor": optimizer_actor.param_groups[0]['weight_decay'],
        "weight_decay_critic": optimizer_critic.param_groups[0]['weight_decay'],
        "epsilon_decay": epsilon_decay,
        "gamma": gamma,
        "patience": patience
    })
    
    # 초기 탐험율 설정
    epsilon = 1.0
    epsilon_min = 0.1

    # 체크포인트 로드 (선택 사항)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 출력 레이어 제외하고 로드
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith('actor_fc.3') and not k.startswith('critic_fc.3'):
                new_state_dict[k] = v

        # 기존 모델의 상태를 업데이트 (strict=False로 설정하여 일치하지 않는 키는 무시)
        policy_net.load_state_dict(new_state_dict, strict=False)

        # 출력 레이어는 새로 초기화
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
        for epoch in tqdm(range(start_epoch, num_epochs + 1), desc="에폭 진행 상황"):
            for batch_idx in range(batch_size):
                state = env.reset()
                done = False
                log_probs = []
                values = []
                rewards = []
                entropy_total = 0.0
                travel_times = []  # 이동 시간 기록

                while not done:
                    positions = state['positions']
                    uavs_info = positions.to(device)
                    action_mask = create_action_mask(state)
                    
                    # 비용 행렬 및 도착 시간 계산
                    cost_matrix = calculate_cost_matrix(positions, env.missions, env.speeds)
                    arrival_times = calculate_arrival_times(positions, env.missions, env.speeds)

                    # 정책 네트워크 통과
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

                    # UAV 선택 순서 결정
                    uav_order = compute_uav_order(env)
                    
                    # 액션 선택
                    actions = choose_action(action_probs, epsilon, uav_order, global_action_mask=action_mask)

                    for i in range(env.num_uavs):
                        action = actions[i]
                        # 액션이 유효한지 확인
                        if not action_mask[action].item():
                            dist = torch.distributions.Categorical(action_probs[i])
                            log_prob = dist.log_prob(torch.tensor(action).to(device))
                            log_probs.append(log_prob)
                            entropy = dist.entropy()
                            entropy_total += entropy
                        else:
                            # 유효하지 않은 액션에 대해서는 로그 확률을 0으로 설정
                            log_probs.append(torch.tensor(0.0).to(device))
                    
                    # 환경 업데이트
                    next_state, reward, done = env.step(actions)

                    rewards.append(reward)
                    values.append(state_values.squeeze())
                    state = next_state

                    # 이동 시간 기록
                    travel_times.append(env.cumulative_travel_times.clone())

                # 누적 보상 계산 (할인율 적용)
                returns = []
                R = 0
                for r in reversed(rewards):
                    R = r + gamma * R
                    returns.insert(0, R)
                returns = torch.tensor(returns, device=device)

                # 표준화 (선택 사항)
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
                    policy_loss_total = torch.stack(policy_loss).mean() - 0.01 * entropy_total.mean()
                    value_loss_total = torch.stack(value_loss).mean()
                    loss = policy_loss_total + value_loss_total
                    
                    # 옵티마이저 초기화
                    optimizer_actor.zero_grad()
                    optimizer_critic.zero_grad()
                    
                    # 손실 역전파
                    loss.backward()
                    
                    # 그래디언트 클리핑
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                    
                    # 옵티마이저 스텝
                    optimizer_actor.step()
                    optimizer_critic.step()

                # 탐험율 감소
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay

                # 보상과 이동 시간 로깅
                average_travel_time = torch.stack(travel_times).mean().item() if travel_times else 0.0

                # 로그 기록
                wandb.log({
                    "episode": episode,
                    "epoch": epoch,
                    "batch": batch_idx,
                    "policy_loss": policy_loss_total.item() if policy_loss else 0,
                    "value_loss": value_loss_total.item() if value_loss else 0,
                    "reward": rewards[-1],
                    "epsilon": epsilon,
                    "entropy": entropy_total.mean().item() if 'entropy_total' in locals() else 0,
                    "average_travel_time": average_travel_time  # 평균 이동 시간 추가
                })

                # 가시화 및 체크포인트 저장
                if episode % 100 == 0:
                    # 체크포인트 저장
                    checkpoint_save_path = os.path.join(checkpoints_path, f"episode_{episode}.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': policy_net.state_dict(),
                        'optimizer_actor_state_dict': optimizer_actor.state_dict(),
                        'optimizer_critic_state_dict': optimizer_critic.state_dict(),
                        'epsilon': epsilon
                    }, checkpoint_save_path)
                    
                    # 시각화 저장
                    visualization_path = os.path.join(results_path, f"2opt_mission_paths_episode_{episode}.png")
                    visualize_results(
                        env, 
                        visualization_path,
                        reward=rewards[-1],
                        epsilon=epsilon,
                        policy_loss=policy_loss_total.item() if policy_loss else 0,
                        value_loss=value_loss_total.item() if value_loss else 0,
                        folder_name='Training Results'
                    )
                    wandb.log({
                        "mission_paths": wandb.Image(visualization_path)
                    })

                episode += 1

            # 검증 수행
            if epoch % 100 == 0:
                best_validation_reward = validate_model(
                    env=val_env, 
                    policy_net=policy_net, 
                    device=device, 
                    edge_index=edge_index, 
                    batch=batch, 
                    checkpoints_best_path=checkpoints_best_path, 
                    images_best_path=images_best_path, 
                    epoch=epoch, 
                    best_validation_reward=best_validation_reward
                )
                
                # 조기 종료는 validate_model에서 처리됨
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
