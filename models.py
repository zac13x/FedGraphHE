"""
HMAGT: Hierarchical Multi-scale Adaptive Graph Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImagePatchGraphConstructor(nn.Module):
    def __init__(self, patch_size=16, overlap=8):
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
    
    def extract_patches(self, x):
        B, C, H, W = x.size()
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)
        
        patches = patches.transpose(1, 2)
        N = patches.size(1)

        num_patches_h = (H - self.patch_size) // self.stride + 1
        num_patches_w = (W - self.patch_size) // self.stride + 1
        
        positions = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                positions.append([i, j])
        positions = torch.tensor(positions, dtype=torch.float32, device=x.device)
        positions = positions.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)
        
        return patches, positions
    
    def build_spatial_edges(self, positions, k_neighbors=8):
        B, N, _ = positions.size()
        edge_index_list = []
        
        for b in range(B):
            pos = positions[b]  # (N, 2)

            dist = torch.cdist(pos, pos, p=2)  # (N, N)

            _, indices = torch.topk(dist, k=min(k_neighbors + 1, N), largest=False)
            indices = indices[:, 1:]

            src = torch.arange(N, device=pos.device).unsqueeze(1).repeat(1, indices.size(1))
            tgt = indices
            edge_index = torch.stack([src.flatten(), tgt.flatten()], dim=0)
            
            edge_index_list.append(edge_index)
        
        return edge_index_list


class ParallelMultiScaleAggregation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W_1hop = nn.Linear(in_channels, out_channels)
        self.W_2hop = nn.Linear(in_channels, out_channels)

        self.W_fusion = nn.Linear(out_channels * 2, out_channels)
        self.gate = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.Sigmoid()
        )
    
    def aggregate_neighbors(self, x, edge_index, hop=1):
        N = x.size(0)
        src, tgt = edge_index

        if hop == 1:
            neighbor_feats = x[tgt]  
        else:
            adj = torch.zeros(N, N, device=x.device)
            adj[src, tgt] = 1
            adj_2hop = torch.matmul(adj, adj)
            adj_2hop = (adj_2hop > 0).float()
            adj_2hop.fill_diagonal_(0)

            src_2hop, tgt_2hop = torch.where(adj_2hop > 0)
            neighbor_feats = x[tgt_2hop]
            src = src_2hop

        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, src, neighbor_feats)
        
        return aggregated
    
    def forward(self, x, edge_index):
        z_1hop = self.aggregate_neighbors(x, edge_index, hop=1)
        z_1hop = self.W_1hop(z_1hop)
        
        z_2hop = self.aggregate_neighbors(x, edge_index, hop=2)
        z_2hop = self.W_2hop(z_2hop)

        multi_scale_feat = torch.cat([z_1hop, z_2hop], dim=-1)
        gate_weight = self.gate(multi_scale_feat)
        
        z_fused = gate_weight * z_1hop + (1 - gate_weight) * z_2hop
        
        return z_fused


class HierarchicalPooling(nn.Module):
    def __init__(self, in_channels, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.score_layer = nn.Linear(in_channels, 1)
    
    def forward(self, x):
        scores = self.score_layer(x).squeeze(-1)  # (N,)
        
        N = x.size(0)
        k = max(1, int(self.ratio * N))
        
        _, idx = torch.topk(scores, k, largest=True)
        x_pooled = x[idx]
        
        return x_pooled, idx


class GraphTransformerLayer(nn.Module):
    
    def __init__(self, in_channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * 4, in_channels)
        )
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x_in = x.unsqueeze(0)
        attn_out, _ = self.attn(x_in, x_in, x_in)
        x_in = self.norm1(x_in + self.dropout(attn_out))

        ff_out = self.ff(x_in)
        x_out = self.norm2(x_in + self.dropout(ff_out))
        
        return x_out.squeeze(0)


class HMAGT(nn.Module):
    def __init__(self, num_classes=2, backbone_channels=64, 
                 hidden_dim=128, k_neighbors=8, pool_ratio=0.5,
                 num_heads=4, dropout=0.1):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, backbone_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(backbone_channels, backbone_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.graph_constructor = ImagePatchGraphConstructor(patch_size=8, overlap=4)

        patch_feature_dim = backbone_channels * 8 * 8
        self.patch_projection = nn.Linear(patch_feature_dim, hidden_dim)

        self.multi_scale_agg = ParallelMultiScaleAggregation(hidden_dim, hidden_dim)

        self.pooling = HierarchicalPooling(hidden_dim, ratio=pool_ratio)

        self.transformer = GraphTransformerLayer(hidden_dim, num_heads, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.k_neighbors = k_neighbors
    
    def forward(self, x):
        B = x.size(0)

        features = self.backbone(x)

        patches, positions = self.graph_constructor.extract_patches(features)
        edge_index_list = self.graph_constructor.build_spatial_edges(positions, self.k_neighbors)

        patches = self.patch_projection(patches)

        outputs = []
        for i in range(B):
            nodes = patches[i]
            edge_index = edge_index_list[i]

            nodes_ms = self.multi_scale_agg(nodes, edge_index)

            nodes_pooled, _ = self.pooling(nodes_ms)

            nodes_refined = self.transformer(nodes_pooled)

            graph_repr = torch.sum(nodes_refined, dim=0)
            outputs.append(graph_repr)
        
        outputs = torch.stack(outputs, dim=0)

        logits = self.classifier(outputs)
        
        return logits

