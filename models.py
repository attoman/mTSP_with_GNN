# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from utils import calculate_total_distance

class EnhancedGNNTransformerEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, heads=8, dropout=0.3):
        super(EnhancedGNNTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
            in_channels = hidden_channels * heads  # 다음 레이어의 입력 채널 수 갱신

        self.gnn_output = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Dropout(dropout)
        )
        
        # 글로벌 self-attention 레이어 추가
        self.global_attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch):
    # GNN 레이어 통과
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)

        x = self.gnn_output(x)  # (num_nodes, out_channels)

        # 그래프별로 분리하여 self-attention 적용
        unique_batches = batch.unique()
        attn_outputs = []
        for b in unique_batches:
            mask = (batch == b)  # 그래프 b에 해당하는 마스크 생성
            x_b = x[mask]  # 해당 그래프의 노드 임베딩
            if x_b.size(0) == 0:
                continue
            x_b = x_b.unsqueeze(1)  # (num_nodes_b, 1, out_channels)
            attn_output, attn_weights = self.global_attention(x_b, x_b, x_b)  # (num_nodes_b, 1, out_channels)
            attn_output = self.layer_norm(attn_output.squeeze(1))  # (num_nodes_b, out_channels)
            attn_output = self.dropout(attn_output)
            attn_outputs.append(attn_output)

        # 원래 노드 순서로 재조립
        attn_outputs = torch.cat(attn_outputs, dim=0)
        return attn_outputs

class ImprovedActorCriticNetwork(nn.Module):
    def __init__(self, num_missions, num_uavs, embedding_dim=64, hidden_dim=128, num_layers=4, heads=8):
        super(ImprovedActorCriticNetwork, self).__init__()
        self.num_missions = num_missions
        self.num_uavs = num_uavs

        total_in_channels = 2 + 1 + 1 + 1 + 1  # 미션 좌표(2) + 마스크 정보(1) + UAV 속도 정보(1) + UAV 비용 정보(1) + 예상 도착 시간(1)

        self.gnn_encoder = EnhancedGNNTransformerEncoder(
            in_channels=total_in_channels,
            hidden_channels=embedding_dim,
            out_channels=embedding_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=0.3
        )

        combined_feature_size = embedding_dim + 2 + 1  # 임베딩 + UAV의 2D 좌표(x, y) + UAV 속도

        # 액터 네트워크
        self.actor_fc = nn.Sequential(
            nn.Linear(combined_feature_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_missions)
        )
        # 크리틱 네트워크
        self.critic_fc = nn.Sequential(
            nn.Linear(combined_feature_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, mission_coords, edge_index, batch, uavs_info, action_mask, speeds, cost_matrix, arrival_times):
        # 마스크 정보 차원 조정 및 임베딩
        mask_embedded = action_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, self.num_uavs, 1).float()  # (num_missions, num_uavs, 1)
        speeds_embedded = speeds.unsqueeze(0).repeat(mission_coords.size(0), 1).unsqueeze(-1).float()  # (num_missions, num_uavs, 1)
        cost_embedded = cost_matrix.T.unsqueeze(-1)  # (num_missions, num_uavs, 1)
        arrival_times_embedded = arrival_times.T.unsqueeze(-1)  # (num_missions, num_uavs, 1)

        # 임베딩 정보 결합
        mission_coords_expanded = mission_coords.unsqueeze(1).repeat(1, self.num_uavs, 1)  # (num_missions, num_uavs, 2)
        combined_embedded = torch.cat([mission_coords_expanded, mask_embedded, speeds_embedded, cost_embedded, arrival_times_embedded], dim=-1)  # (num_missions, num_uavs, feature_num)

        # 텐서 크기 재조정
        combined_embedded = combined_embedded.view(-1, combined_embedded.size(-1))  # (num_missions * num_uavs, feature_num)

        # batch 텐서 재조정: 각 UAV에 대해 num_missions만큼 반복된 배치 인덱스 생성
        new_batch = batch.repeat_interleave(self.num_uavs)  # (num_missions * num_uavs)

        # GNN 인코더 통과
        mission_embeddings = self.gnn_encoder(combined_embedded, edge_index, new_batch)  # (num_missions * num_uavs, embedding_dim)

        # 임베딩 차원 재조정
        mission_embeddings_expanded = mission_embeddings.view(mission_coords.size(0), self.num_uavs, -1)  # (num_missions, num_uavs, embedding_dim)
        mission_embeddings_expanded = mission_embeddings_expanded.permute(1, 0, 2).contiguous().view(-1, mission_embeddings_expanded.size(2))  # (num_uavs * num_missions, embedding_dim)

        # UAV 정보 및 임베딩 결합
        uavs_info_repeated = uavs_info.repeat(mission_coords.size(0), 1)  # (num_uavs * num_missions, 2)
        speeds_repeated = speeds.repeat(mission_coords.size(0)).unsqueeze(-1)  # (num_uavs * num_missions, 1)

        combined = torch.cat([
            uavs_info_repeated,  # (num_uavs * num_missions, 2)
            mission_embeddings_expanded,  # (num_uavs * num_missions, embedding_dim)
            speeds_repeated  # (num_uavs * num_missions, 1)
        ], dim=-1)  # (num_uavs * num_missions, 2 + embedding_dim + 1)

        # 액터와 크리틱 네트워크 통과
        action_logits = self.actor_fc(combined)
        action_probs = F.softmax(action_logits, dim=-1)
        state_values = self.critic_fc(combined)

        return action_probs, state_values
