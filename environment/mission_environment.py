# environment/mission_environment.py

import torch
from utils.calculations import calculate_distance, calculate_travel_time
from utils.masks import create_action_mask

class MissionEnvironment:
    """
    다중 UAV 미션 할당을 위한 강화 학습 환경 클래스.
    """
    def __init__(self, missions=None, uavs_start=None, uavs_speeds=None, device='cpu', mode='train', seed=None, time_weight=2.0):
        self.device = device
        self.mode = mode
        self.seed = seed
        self.num_missions = missions.size(0) if missions is not None else 20
        self.num_uavs = uavs_start.size(0) if uavs_start is not None else 3

        self.missions = missions
        self.uavs_start = uavs_start
        self.speeds = uavs_speeds
        self.time_weight = time_weight
        self.reset()

    def reset(self):
        """환경을 초기 상태로 리셋합니다."""
        self.current_positions = self.uavs_start.clone()
        self.visited = torch.zeros(self.num_missions, dtype=torch.bool, device=self.device)
        self.reserved = torch.zeros(self.num_missions, dtype=torch.bool, device=self.device)
        self.paths = [[] for _ in range(self.num_uavs)]
        self.cumulative_travel_times = torch.zeros(self.num_uavs, device=self.device)
        self.ready_for_next_action = torch.ones(self.num_uavs, dtype=torch.bool, device=self.device)
        self.targets = [-1] * self.num_uavs
        self.remaining_distances = torch.full((self.num_uavs,), float('inf'), device=self.device)

        self.visited[0] = True  # 시작 미션을 방문한 것으로 설정

        for i in range(self.num_uavs):
            self.paths[i].append(0)  # 각 UAV의 경로에 시작 미션 추가
        return self.get_state()

    def get_state(self):
        """현재 환경 상태를 반환합니다."""
        return {
            'positions': self.current_positions.clone(),
            'visited': self.visited.clone(),
            'reserved': self.reserved.clone(),
            'ready_for_next_action': self.ready_for_next_action.clone(),
            'remaining_distances': self.remaining_distances.clone(),
            'targets': self.targets
        }

    def step(self, actions):
        """
        액션을 실행하고 환경 상태를 업데이트합니다.
        
        Args:
            actions (list): UAV들이 선택한 액션.
        
        Returns:
            tuple: 다음 상태, 소요 시간 텐서, 종료 여부.
        """
        # 보상은 외부에서 계산하도록 환경에서는 패널티를 누적하지 않습니다.
        # 보상 계산은 학습 루프에서 별도로 처리됩니다.
        for i, action in enumerate(actions):
            if self.ready_for_next_action[i] and not self.visited[action] and not self.reserved[action]:
                self.reserved[action] = True
                self.ready_for_next_action[i] = False
                self.targets[i] = action
                mission_from = self.current_positions[i]
                mission_to = self.missions[action]
                self.remaining_distances[i] = calculate_distance(mission_from, mission_to)

        # 보상 계산을 위해 기록
        travel_times = torch.zeros(self.num_uavs, device=self.device)

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

                travel_times[i] = travel_time

        done = self.visited.all()

        # 모든 임무가 완료된 후 UAV들이 반드시 출발 지점으로 돌아가도록 처리
        if done:
            for i in range(self.num_uavs):
                if self.targets[i] != 0:  # 이미 출발 지점에 있지 않은 경우
                    self.targets[i] = 0
                    self.ready_for_next_action[i] = False
                    self.remaining_distances[i] = calculate_distance(self.current_positions[i], self.missions[0])

        return self.get_state(), travel_times, done
