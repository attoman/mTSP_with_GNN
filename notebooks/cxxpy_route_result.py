import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cvxpy as cp
import torch
from itertools import combinations
import os
from datetime import datetime

# ============================
# Utility Functions
# ============================

def calculate_distance(mission1, mission2):
    return torch.sqrt(torch.sum((mission1 - mission2) ** 2))

def calculate_travel_time(distance, speed):
    return distance / speed

def create_edge_index(num_missions):
    adj_matrix = torch.ones((num_missions, num_missions)) - torch.eye(num_missions)
    edge_index = torch.nonzero(adj_matrix).t()
    return edge_index

def create_action_mask(state):
    visited = state['visited']
    reserved = state['reserved']
    action_mask = visited | reserved
    return action_mask

# ============================
# 2-opt Algorithm for Path Optimization
# ============================

def two_opt(route, missions, end_point=None):
    """
    2-opt 알고리즘을 사용하여 경로를 최적화합니다.
    종료 지점이 주어지면 마지막 지점을 고정합니다.
    """
    best_route = route[:]
    best_distance = calculate_total_distance(best_route, missions, end_point)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                new_distance = calculate_total_distance(new_route, missions, end_point)
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
        route = best_route[:]

    return best_route

def calculate_total_distance(route, missions, end_point=None):
    distance = 0.0
    for i in range(len(route) - 1):
        distance += calculate_distance(missions[route[i]], missions[route[i + 1]]).item()
    if end_point is not None:
        # 마지막 지점에서 종료 지점까지의 거리 추가
        distance += np.linalg.norm(end_point - missions[route[-1]].cpu().numpy())
    return distance

# ============================
# Data Class
# ============================

class MissionData:
    def __init__(self, num_missions=20, num_uavs=3, seed=42, device='cpu'):
        self.num_missions = num_missions
        self.num_uavs = num_uavs
        self.seed = seed
        self.device = device
        self.missions, self.uavs_start, self.uavs_end, self.uavs_speeds = self.generate_data()

    def generate_data(self):
        torch.manual_seed(self.seed)
        # 미션 생성 (시작 지점 포함)
        missions = torch.rand((self.num_missions, 2)) * 100
        start_mission = missions[0].unsqueeze(0)  # 미션 0은 시작 지점
        # 공통 종료 지점 생성 (시작 지점과 다르게 설정)
        while True:
            end_mission = torch.rand((1, 2)) * 100
            if not torch.equal(start_mission, end_mission):
                break
        uavs_end = end_mission.repeat(self.num_uavs, 1)  # 모든 UAV가 동일한 종료 지점
        # UAV의 시작 위치 (모두 시작 지점)
        uavs_start = start_mission.repeat(self.num_uavs, 1)
        # UAV의 속도
        uavs_speeds = torch.rand(self.num_uavs) * 9 + 1  # 속도 범위: 1 ~ 10
        return missions.to(self.device), uavs_start.to(self.device), uavs_end.to(self.device), uavs_speeds.to(self.device)

# ============================
# Reinforcement Learning Environment Class (MARL)
# ============================

class MissionEnvironment:
    def __init__(self, missions, uavs_start, uavs_end, uavs_speeds, device):
        self.missions = missions
        self.uavs_end = uavs_end  # 모든 UAV이 공유하는 종료 지점
        self.num_missions = missions.size(0)
        self.num_uavs = uavs_start.size(0)
        self.speeds = uavs_speeds
        self.uavs_start = uavs_start
        self.device = device
        self.reset()

    def reset(self):
        self.current_positions = self.uavs_start.clone()
        self.visited = torch.zeros(self.num_missions, dtype=torch.bool, device=self.device)
        self.reserved = torch.zeros(self.num_missions, dtype=torch.bool, device=self.device)
        self.paths = [[] for _ in range(self.num_uavs)]
        self.cumulative_travel_times = torch.zeros(self.num_uavs, device=self.device)
        self.ready_for_next_action = torch.ones(self.num_uavs, dtype=torch.bool, device=self.device)
        self.targets = [-1] * self.num_uavs
        self.remaining_distances = torch.full((self.num_uavs,), float('inf'), device=self.device)
        for i in range(self.num_uavs):
            self.paths[i].append(0)  # 시작 지점 추가

        # 최적화 수행
        optimized_routes, total_travel_times = self.optimize_routes()

        # 최적화된 경로 할당
        for i in range(self.num_uavs):
            if optimized_routes[i]:
                self.paths[i] = optimized_routes[i]
                self.cumulative_travel_times[i] = total_travel_times[i]

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
                travel_time = calculate_travel_time(distance, self.speeds[i])

                self.cumulative_travel_times[i] += travel_time
                self.current_positions[i] = self.missions[action]
                self.visited[action] = True
                self.paths[i].append(action)
                self.ready_for_next_action[i] = True
                self.reserved[action] = False

        done = self.visited.all()
        if done:
            for i in range(self.num_uavs):
                if not torch.equal(self.current_positions[i], self.uavs_end[i]):
                    distance = calculate_distance(self.current_positions[i], self.uavs_end[i])
                    travel_time = calculate_travel_time(distance, self.speeds[i])
                    self.cumulative_travel_times[i] += travel_time
                    self.current_positions[i] = self.uavs_end[i]
                    end_mission = self.num_missions  # 종료 지점을 새로운 미션으로 간주
                    self.paths[i].append(end_mission)
            for i in range(self.num_uavs):
                # 2-opt 알고리즘을 사용한 경로 최적화 (종료 지점 고정)
                self.paths[i] = two_opt(self.paths[i], self.missions, end_point=self.uavs_end[i].cpu().numpy())
                    
            total_travel_time = self.cumulative_travel_times.max().item()
            reward = -total_travel_time
        else:
            reward = 0.0

        return self.get_state(), reward, done

    def optimize_routes(self):
        # Tensor를 NumPy 배열로 변환
        missions_np = self.missions.cpu().numpy()
        uavs_start_np = self.uavs_start.cpu().numpy()
        uavs_end_np = self.uavs_end.cpu().numpy()
        uavs_speeds_np = self.speeds.cpu().numpy()

        num_missions = self.num_missions
        num_uavs = self.num_uavs

        # 거리 행렬 계산
        distance_matrix = calculate_distance_matrix(missions_np)

        # 변수 정의
        x = cp.Variable((num_uavs, num_missions), boolean=True)
        T = cp.Variable(num_uavs)

        # 제약 조건 설정
        constraints = []

        # 각 미션은 정확히 하나의 UAV에 할당 (시작 미션 제외, 종료 지점은 별도로 처리)
        for j in range(1, num_missions):  # 미션 0은 시작, 미션 num_missions-1은 종료 (종료는 별도로 처리)
            constraints.append(cp.sum(x[:, j]) == 1)

        # UAV는 시작 지점에 할당
        for i in range(num_uavs):
            constraints.append(x[i, 0] == 1)  # 시작 미션

        # UAV는 최소 한 개의 미션을 수행
        for i in range(num_uavs):
            constraints.append(cp.sum(x[i, 1:num_missions]) >= 1)  # 최소 한 개의 미션 할당

        # 이동 시간 계산: 각 UAV의 총 이동 시간을 T[i]로 제한
        for i in range(num_uavs):
            # 할당된 미션들과 시작 지점 간의 거리 합을 계산
            travel_time = cp.sum(cp.multiply(distance_matrix[0][1:num_missions], x[i,1:num_missions])) / uavs_speeds_np[i]
            # 종료 지점까지의 이동 시간 추가
            # UAV의 마지막 미션과 종료 지점 간의 거리를 정확히 반영하려면 추가적인 변수 및 제약 조건이 필요하지만,
            # 여기서는 간단히 시작 지점과 종료 지점 간의 거리로 근사화합니다.
            end_travel_time = cp.norm(cp.hstack([uavs_end_np[i] - self.missions[0].cpu().numpy()]), 2) / uavs_speeds_np[i]
            constraints.append(T[i] >= (travel_time + end_travel_time))

        # 목적 함수: 모든 UAV의 최대 이동 시간을 최소화
        objective = cp.Minimize(cp.max(T))

        # 솔버 선택: Mosek 사용
        try:
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.MOSEK)
            solver_used = 'Mosek'
        except cp.error.SolverError:
            print("Mosek 솔버를 사용할 수 없습니다. 다른 솔버를 시도합니다.")
            try:
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.ECOS_BB)
                solver_used = 'ECOS_BB'
            except cp.error.SolverError:
                print("ECOS_BB 솔버도 사용할 수 없습니다. 문제를 해결할 수 없습니다.")
                return [[] for _ in range(num_uavs)], [0.0 for _ in range(num_uavs)]

        # 결과 확인
        if problem.status not in ["infeasible", "unbounded"]:
            assignment = x.value
            assignment = (assignment > 0.5).astype(int)

            # 각 UAV의 경로 추출
            routes = []
            for i in range(num_uavs):
                route = [0]  # 시작점
                for j in range(1, num_missions):
                    if assignment[i, j]:
                        route.append(j)
                # 종료 지점 추가 (공통 종료 지점의 인덱스는 num_missions)
                end_mission = num_missions  # 종료 지점은 missions_np에 포함되지 않으므로, 인덱스를 num_missions으로 설정
                route.append(end_mission)
                # 종료 지점의 좌표는 missions_np[-1]과 다르므로, missions_np를 수정하여 종료 지점 좌표를 추가
                missions_extended = np.vstack([missions_np, uavs_end_np[i]])
                optimized_route = two_opt(route, torch.tensor(missions_extended), end_point=uavs_end_np[i])
                routes.append(optimized_route)

            # 총 이동 시간 계산
            total_travel_times = np.zeros(num_uavs)
            for i in range(num_uavs):
                for k in range(len(routes[i]) - 1):
                    from_mission = routes[i][k]
                    to_mission = routes[i][k + 1]
                    if to_mission == num_missions:
                        # 종료 지점
                        to_coords = uavs_end_np[i]
                    else:
                        to_coords = missions_np[to_mission]
                    if from_mission == num_missions:
                        # 종료 지점에서 시작
                        from_coords = uavs_end_np[i]
                    else:
                        from_coords = missions_np[from_mission]
                    distance = np.linalg.norm(from_coords - to_coords)
                    total_travel_times[i] += distance / uavs_speeds_np[i]

            print(f"사용된 솔버: {solver_used}")
            return routes, total_travel_times
        else:
            print("문제가 해결되지 않았습니다:", problem.status)
            return [[] for _ in range(num_uavs)], [0.0 for _ in range(num_uavs)]

def calculate_distance_matrix(missions):
    num_missions = missions.shape[0]
    dist_matrix = np.zeros((num_missions, num_missions))
    for i in range(num_missions):
        for j in range(num_missions):
            if i != j:
                dist_matrix[i][j] = np.linalg.norm(missions[i] - missions[j])
            else:
                dist_matrix[i][j] = 0
    return dist_matrix

# ============================
# Visualization Function
# ============================

def visualize_routes(missions, routes, uavs_end, uavs_speeds, num_uavs, num_missions):
    plt.figure(figsize=(10, 8))
    colors = cm.get_cmap('tab10', num_uavs)

    # 스타트 포인트 플롯 (미션 0)
    start_coords = missions[0]
    plt.scatter(start_coords[0], start_coords[1], c='blue', marker='X', s=100, label='Start Point')
    plt.text(start_coords[0] + 0.5, start_coords[1] + 0.5, "Start Point", fontsize=9, color='blue')

    # 미션 위치 플롯 (미션 1부터)
    plt.scatter(missions[1:, 0], missions[1:, 1], c='black', marker='o', label='Missions')
    for idx, (x, y) in enumerate(missions[1:], start=1):
        plt.text(x + 0.5, y + 0.5, str(idx), fontsize=9)

    # 종료 지점 플롯 (공통 종료 지점이므로 한 번만 표시)
    unique_end = uavs_end[0]  # 모든 UAV의 종료 지점이 동일하므로 첫 번째 UAV의 종료 지점 사용
    plt.scatter(unique_end[0], unique_end[1], c='blue', marker='X', s=100, label='End Point')  # 같은 형식으로 표시
    plt.text(unique_end[0] + 0.5, unique_end[1] + 0.5, "End Point", fontsize=9, color='blue')

    # 각 UAV의 경로 플롯
    for i in range(num_uavs):
        route = routes[i]
        if len(route) < 2:
            continue
        # 종료 지점은 num_missions 인덱스로 설정되었으므로, 이를 제외하고 경로를 그립니다.
        route_coords = []
        for point in route:
            if point == num_missions:
                # 종료 지점
                route_coords.append(unique_end)
            else:
                route_coords.append(missions[point])
        route_coords = np.array(route_coords)
        # UAV의 속도를 레이블에 포함
        plt.plot(route_coords[:, 0], route_coords[:, 1], marker='o', color=colors(i), label=f'UAV {i+1} (Speed: {uavs_speeds[i]:.2f})')

    plt.title('mosek_cvxpy_optimize')
    plt.xlabel('X 좌표')
    plt.ylabel('Y 좌표')
    plt.legend()
    plt.grid(True)
    
    # 이미지 파일 저장
    try:
        # 현재 스크립트의 디렉토리 경로 가져오기
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # __file__ 변수가 정의되지 않은 경우 현재 작업 디렉토리 사용
            script_dir = os.getcwd()
        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 파일 이름 지정
        filename = f"uav_routes_{timestamp}.png"
        # 전체 경로 생성
        filepath = os.path.join(script_dir, filename)
        # 이미지 저장
        plt.savefig(filepath)
        print(f"경로 시각화 이미지가 '{filepath}'에 저장되었습니다.")
    except Exception as e:
        print(f"이미지 저장 중 오류가 발생했습니다: {e}")
    
    # 시각화 표시
    plt.show()

# ============================
# 실행 예제
# ============================

if __name__ == "__main__":
    # 데이터 초기화
    mission_data = MissionData(num_missions=10, num_uavs=3, seed=42, device='cpu')  # 미션 수를 줄여 계산 시간을 단축
    missions = mission_data.missions
    uavs_start = mission_data.uavs_start
    uavs_end = mission_data.uavs_end
    uavs_speeds = mission_data.uavs_speeds

    # 환경 초기화
    env = MissionEnvironment(missions, uavs_start, uavs_end, uavs_speeds, device='cpu')

    # 초기 상태
    state = env.reset()

    # 결과 출력
    print("최적화된 UAV 경로 및 총 이동 시간:")
    for i in range(env.num_uavs):
        print(f"UAV {i+1} 경로: {env.paths[i]}")
        print(f"UAV {i+1} 총 이동 시간: {env.cumulative_travel_times[i].item():.2f}")

    # 시각화
    routes = env.paths
    missions_np = missions.cpu().numpy()
    visualize_routes(missions_np, routes, uavs_end.cpu().numpy(), uavs_speeds.cpu().numpy(), env.num_uavs, env.num_missions)
