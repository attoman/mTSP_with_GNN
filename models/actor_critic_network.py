# models/actor_critic_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gnn_transformer_encoder import EnhancedGNNTransformerEncoder

class ImprovedActorCriticNetwork(nn.Module):
    """
    Enhanced GNN Transformer 인코더를 사용하는 액터-크리틱 네트워크.
    """
    def __init__(self, num_missions, num_uavs, embedding_dim=64, hidden_dim=128, num_layers=4, heads=8,
                 gnn_dropout=0.3, actor_dropout=0.3, critic_dropout=0.3):
        super(ImprovedActorCriticNetwork, self).__init__()
        self.num_missions = num_missions
        self.num_uavs = num_uavs

        # 총 입력 채널: 미션 좌표(2) + 마스크(1) + UAV 속도(1) + 비용(1) + 예상 도착 시간(1)
        total_in_channels = 2 + 1 + 1 + 1 + 1

        self.gnn_encoder = EnhancedGNNTransformerEncoder(
            in_channels=total_in_channels,
            hidden_channels=embedding_dim,
            out_channels=embedding_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=gnn_dropout
        )

        # 결합된 특성 크기: 임베딩 + UAV의 2D 좌표 + UAV 속도
        self.combined_feature_size = embedding_dim + 2 + 1

        # 액터 네트워크
        self.actor_fc = nn.Sequential(
            nn.Linear(self.combined_feature_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(actor_dropout),
            nn.Linear(hidden_dim, num_missions)
        )

        # 크리틱 네트워크
        self.critic_fc = nn.Sequential(
            nn.Linear(self.combined_feature_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(critic_dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, mission_coords, edge_index, batch, uavs_info, action_mask, speeds, cost_matrix, arrival_times):
        """
        액터-크리틱 네트워크를 통한 순전파.
        """
        # GNN 인코더를 위한 입력 전처리
        mask_embedded = action_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, self.num_uavs, 1).float()
        speeds_embedded = speeds.unsqueeze(0).repeat(mission_coords.size(0), 1).unsqueeze(-1).float()
        cost_embedded = cost_matrix.T.unsqueeze(-1)
        arrival_times_embedded = arrival_times.T.unsqueeze(-1)
        
        mission_coords_expanded = mission_coords.unsqueeze(1).repeat(1, self.num_uavs, 1)
        combined_embedded = torch.cat([mission_coords_expanded, mask_embedded, speeds_embedded, cost_embedded, arrival_times_embedded], dim=-1)
        combined_embedded = combined_embedded.view(-1, combined_embedded.size(-1))
        
        new_batch = batch.repeat_interleave(self.num_uavs)
        
        mission_embeddings = self.gnn_encoder(combined_embedded, edge_index, new_batch)
        
        mission_embeddings_expanded = mission_embeddings.view(mission_coords.size(0), self.num_uavs, -1)
        mission_embeddings_expanded = mission_embeddings_expanded.permute(1, 0, 2).contiguous().view(-1, mission_embeddings_expanded.size(2))
        
        uavs_info_repeated = uavs_info.repeat(mission_coords.size(0), 1)
        speeds_repeated = speeds.repeat(mission_coords.size(0)).unsqueeze(-1)
        
        combined = torch.cat([
            uavs_info_repeated,
            mission_embeddings_expanded,
            speeds_repeated
        ], dim=-1)
        
        # 결합된 특성 크기가 네트워크에 맞도록 조정
        n_features = combined.size(-1)
        combined = combined.view(-1, n_features)
        
        # 액터와 크리틱 네트워크 순전파
        action_logits = self.actor_fc(combined)
        action_probs = F.softmax(action_logits, dim=-1)
        state_values = self.critic_fc(combined)
        
        return action_probs, state_values
