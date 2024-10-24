# environment/mission_environment.py

import torch
from utils.calculations import calculate_distance, calculate_travel_time
from utils.masks import create_action_mask

class MissionEnvironment:
    """
    다중 UAV 미션 할당을 위한 강화 학습 환경 클래스.
    """
    def __init__(self, missions=None, uavs_start=None, uavs_speeds=None, device='cpu', mode='train', seed=None, time_weight=2.0, use_2opt=False):
        self.device = device
        self.mode = mode
        self.seed = seed
        self.num_missions = missions.size(0) if missions is not None else 20
        self.num_uavs = uavs_start.size(0) if uavs_start is not None else 3

        self.missions = missions
        self.uavs_start = uavs_start
        self.speeds = uavs_speeds
        self.use_2opt = use_2opt  # use_2opt 인자를 저장
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

    def create_action_mask(self, state):
        """
        방문한 미션과 예약된 미션을 기반으로 액션 마스크를 생성합니다.
        
        Args:
            state (dict): 현재 상태로 'visited'와 'reserved' 텐서를 포함합니다.
            
        Returns:
            torch.Tensor: 액션 마스크 텐서.
        """
        visited = state['visited']
        reserved = state['reserved']
        action_mask = visited | reserved

        # 시작점(0번 미션)은 임무 도중에는 방문할 수 없도록 설정
        action_mask[0] = True
        
        # 모든 미션(시작점 제외)이 방문되지 않은 경우 시작점 마스크 해제
        if not (visited[1:].all()):
            action_mask[0] = False
        
        return action_mask

    def step(self, actions):
        """
        액션을 실행하고 환경 상태를 업데이트합니다.
        
        Args:
            actions (list): UAV들이 선택한 액션.
        
        Returns:
            tuple: 다음 상태, 소요 시간 텐서, 종료 여부.
        """
        for i, action in enumerate(actions):
            if self.ready_for_next_action[i] and not self.visited[action] and not self.reserved[action]:
                self.reserved[action] = True
                self.ready_for_next_action[i] = False
                self.targets[i] = action
                mission_from = self.current_positions[i]
                mission_to = self.missions[action]
                self.remaining_distances[i] = calculate_distance(mission_from, mission_to)

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