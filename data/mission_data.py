# data/mission_data.py

import torch
import random
import numpy as np

class MissionData:
    """
    미션 데이터를 생성하고 관리하는 클래스.
    """
    def __init__(self, num_missions=20, num_uavs=3, seed=None, device='cpu'):
        self.num_missions = num_missions
        self.num_uavs = num_uavs
        self.seed = seed
        self.device = device
        self.missions, self.uavs_start, self.uavs_speeds = self.generate_data()

    def generate_data(self):
        """랜덤 미션 좌표, UAV 시작 위치, 속도를 생성합니다."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
        else:
            seed = torch.randint(0, 10000, (1,)).item()
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        missions = torch.rand((self.num_missions, 2)) * 100
        missions[-1] = missions[0]  # 마지막 미션을 시작 미션과 동일하게 설정 (시작점과 도착점이 같음)
        start_mission = missions[0].unsqueeze(0)
        uavs_start = start_mission.repeat(self.num_uavs, 1)
        uavs_speeds = torch.rand(self.num_uavs) * 9 + 1  # 속도는 1에서 10 사이
        return missions.to(self.device), uavs_start.to(self.device), uavs_speeds.to(self.device)

    def reset_data(self, seed=None):
        """새로운 시드를 사용하여 미션 데이터를 재설정합니다."""
        self.seed = seed
        self.missions, self.uavs_start, self.uavs_speeds = self.generate_data()
