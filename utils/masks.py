# utils/masks.py

import torch

def create_edge_index(num_missions, num_uavs):
    """
    각 UAV에 대해 모든 가능한 미션 경로를 연결하는 edge_index를 생성합니다.
    """
    edge_index = []
    for u in range(num_uavs):
        base = u * num_missions
        for m1 in range(num_missions):
            for m2 in range(num_missions):
                if m1 != m2:
                    edge_index.append([base + m1, base + m2])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

def create_action_mask(state):
    """
    방문한 미션과 예약된 미션을 기반으로 액션 마스크를 생성합니다.
    """
    visited = state['visited']
    reserved = state['reserved']
    action_mask = visited | reserved
    # 모든 미션(시작점 제외)이 방문되지 않은 경우 시작점 마스크
    if not (visited[1:].all()):
        action_mask[0] = True
    return action_mask
