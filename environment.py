# environment.py

import torch
from utils import calculate_distance, calculate_travel_time, two_opt, calculate_total_distance, create_action_mask
from data import MissionData

class MissionEnvironment:
    def __init__(self, missions=None, uavs_start=None, uavs_speeds=None, device='cpu', mode='train', seed=None, time_weight=2.0):
        self.device = device
        self.mode = mode
        self.seed = seed
        self.num_missions = missions.size(0) if missions is not None else 20
        self.num_uavs = uavs_start.size(0) if uavs_start is not None else 3
        
        self.mission_data = MissionData(
            num_missions=self.num_missions,
            num_uavs=self.num_uavs,
            seed=self.seed,
            device=self.device
        )
        
        self.missions = self.mission_data.missions
        self.uavs_start = self.mission_data.uavs_start
        self.speeds = self.mission_data.uavs_speeds
        self.time_weight = time_weight
        self.reset()

    def reset(self):
        # 새로운 미션 데이터 생성
        self.mission_data.reset_data()
        self.missions = self.mission_data.missions
        self.uavs_start = self.mission_data.uavs_start
        self.speeds = self.mission_data.uavs_speeds
        
        self.current_positions = self.uavs_start.clone()
        self.visited = torch.zeros(self.num_missions, dtype=torch.bool, device=self.device)
        self.reserved = torch.zeros(self.num_missions, dtype=torch.bool, device=self.device)
        self.paths = [[] for _ in range(self.num_uavs)]
        self.cumulative_travel_times = torch.zeros(self.num_uavs, device=self.device)
        self.ready_for_next_action = torch.ones(self.num_uavs, dtype=torch.bool, device=self.device)
        self.targets = [-1] * self.num_uavs
        self.remaining_distances = torch.full((self.num_uavs,), float('inf'), device=self.device)
        
        self.visited[0] = True  # 출발 미션을 방문한 것으로 설정
        
        for i in range(self.num_uavs):
            self.paths[i].append(0)  # 모든 UAV의 경로에 출발 미션 추가
        return self.get_state()

    def get_state(self):
        return {
            'positions': self.current_positions.clone(),
            'visited': self.visited.clone(),
            'reserved': self.reserved.clone(),
            'ready_for_next_action': self.ready_for_next_action.clone(),
            'remaining_distances': self.remaining_distances.clone(),
            'targets': self.targets
        }

    def step(self, actions):
        reward = 0.0
        for i, action in enumerate(actions):
            if self.ready_for_next_action[i] and not self.visited[action] and not self.reserved[action]:
                self.reserved[action] = True
                self.ready_for_next_action[i] = False
                self.targets[i] = action
                mission_from = self.current_positions[i]
                mission_to = self.missions[action]
                self.remaining_distances[i] = calculate_distance(mission_from, mission_to)

        for i, action in enumerate(self.targets):
            if action != -1 and not self.ready_for_next_action[i]:
                distance = self.remaining_distances[i]
                travel_time = calculate_travel_time(distance, self.speeds[i].item())

                self.cumulative_travel_times[i] += travel_time
                self.current_positions[i] = self.missions[action]
                self.visited[action] = True
                self.paths[i].append(action)
                self.ready_for_next_action[i] = True
                self.reserved[action] = False

                # 이동 거리에 대한 보상 패널티
                reward -= distance
                # 이동 시간에 대한 보상 패널티 (가중치 적용)
                reward -= self.time_weight * travel_time

        done = self.visited.all()
        if done:
            for i in range(self.num_uavs):
                if not torch.equal(self.current_positions[i], self.missions[-1]):
                    distance = calculate_distance(self.current_positions[i], self.missions[-1])
                    travel_time = calculate_travel_time(distance, self.speeds[i].item())
                    self.cumulative_travel_times[i] += travel_time
                    self.current_positions[i] = self.missions[-1]
                    self.paths[i].append(self.num_missions - 1)
                    reward -= distance
                    reward -= self.time_weight * travel_time
            total_travel_time = self.cumulative_travel_times.sum().item()
            reward -= self.time_weight * total_travel_time

            if self.mode == 'train':
                # 2-opt 최적화 적용 (시작과 끝 고정)
                optimized_paths = []
                optimized_travel_times = torch.zeros(self.num_uavs, device=self.device)
                for i in range(self.num_uavs):
                    path = self.paths[i]
                    optimized_path = two_opt(path, self.missions)
                    optimized_paths.append(optimized_path)
                    optimized_travel_times[i] = calculate_total_distance(optimized_path, self.missions)
                optimized_total_travel_time = optimized_travel_times.max().item()
                optimized_reward = -optimized_total_travel_time
                # reward = optimized_reward  # 제거

        return self.get_state(), reward, done
