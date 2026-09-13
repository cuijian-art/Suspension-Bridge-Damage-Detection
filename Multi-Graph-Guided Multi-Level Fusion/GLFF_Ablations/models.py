import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# ====================== 你原有的CAM模块完全保留 ======================
class CAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# ====================== 新增：GLFF全局局部特征融合模块 ======================
class GraphMLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()

    def forward(self, x, adj):
        x_agg = adj @ x
        x = self.linear(x_agg)
        x = self.norm(x)
        x = self.act(x)
        return x

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert self.head_dim * num_heads == out_dim, "输出维度必须是头数的整数倍"

        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim)
        )

    def forward(self, x, adj):
        B, D = x.shape
        residual = x
        q = self.q_proj(x).view(B, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(x).view(B, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(x).view(B, self.num_heads, self.head_dim).transpose(0, 1)

        attn = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        mask = (1 - adj.unsqueeze(0)) * (-1e9)
        attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(0, 1).contiguous().view(B, -1)
        out = self.out_proj(out)

        x = self.norm1(residual + out)
        x = self.norm2(x + self.mlp(x))
        return x

class GLFFModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=3, num_heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.local_layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        for _ in range(num_layers):
            self.local_layers.append(GraphMLPLayer(hidden_dim, hidden_dim))
            self.global_layers.append(GraphTransformerLayer(hidden_dim, hidden_dim, num_heads))
        self.graph_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, adj):
        x = self.input_proj(x)
        layer_feats = []
        for i in range(self.num_layers):
            local_feat = self.local_layers[i](x, adj)
            global_feat = self.global_layers[i](x, adj)
            x = local_feat + global_feat
            layer_feats.append(x)
        all_feats = torch.stack(layer_feats, dim=-1)
        pooled_feat = self.graph_pool(all_feats).squeeze(-1)
        return pooled_feat

# ====================== 原有模型仅替换GCN部分为GLFF，其他完全不变 ======================
class MultiLevelFusionModel(nn.Module):
    def __init__(self, n_sensor=6, num_classes=6, window_size=1024):
        super().__init__()
        self.n_sensor = n_sensor
        self.num_classes = num_classes
        # 原有CNN参数完全对齐你原来的设置
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25, stride=1, padding=12),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(32, 24, kernel_size=13, stride=1, padding=6),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(24, 16, kernel_size=13, stride=1, padding=6),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.AdaptiveAvgPool1d(1)
        )
        self.cnn_out_dim = 16
        self.cam = CAM(in_channels=self.cnn_out_dim)

        # 替换原有3层GCNConv为GLFF，输出维度和原来一致，后续逻辑无需修改
        self.glff = GLFFModule(
            in_dim=self.cnn_out_dim,
            hidden_dim=16,  # 和原有GCN输出维度一致，完美兼容
            num_layers=3,
            num_heads=4
        )
        self.gcn_out_dim = 16  # 维度和原来完全一致，不用改后面的层

        # 原有注意力融合、分类头完全保留
        self.attn_fc = nn.Sequential(
            nn.Linear(self.cnn_out_dim + self.gcn_out_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )
        self.attn_softmax = nn.Softmax(dim=1)
        self.cnn_cls = nn.Linear(self.cnn_out_dim, num_classes)
        self.gcn_cls = nn.Linear(self.gcn_out_dim, num_classes)

    def forward(self, x, edge_index, edge_weight, n_per_sensor):
        batch_size = x.shape[0]
        # 原有CNN+CAM提取特征逻辑完全不变
        cnn_feat_list = []
        cnn_logits_list = []
        for s in range(self.n_sensor):
            s_x = x[:, s:s+1, :]
            s_feat = self.cnn(s_x).squeeze(-1)
            s_feat_cam = self.cam(s_feat.unsqueeze(-1)).squeeze(-1)
            cnn_feat_list.append(s_feat_cam)
            cnn_logits_list.append(self.cnn_cls(s_feat_cam))
        cnn_feat = torch.stack(cnn_feat_list, dim=1)
        cnn_logits = torch.stack(cnn_logits_list, dim=1)

        # 新增：把稀疏edge_index转成稠密邻接矩阵供GLFF使用
        total_nodes = batch_size * self.n_sensor
        adj = torch.zeros(total_nodes, total_nodes, device=x.device)
        adj[edge_index[0], edge_index[1]] = edge_weight

        # 替换原有GCN前向为GLFF，输出形状和原来完全一致
        gcn_feat = cnn_feat.reshape(-1, self.cnn_out_dim)
        gcn_feat = self.glff(gcn_feat, adj)
        gcn_feat = gcn_feat.reshape(batch_size, self.n_sensor, self.gcn_out_dim)
        gcn_logits = self.gcn_cls(gcn_feat)

        # 原有注意力融合逻辑完全不变
        concat_feat = torch.cat([cnn_feat, gcn_feat], dim=-1)
        attn_score = self.attn_fc(concat_feat)
        attn_weight = self.attn_softmax(attn_score)
        fused_feat = (attn_weight * concat_feat).flatten(1)

        return fused_feat, cnn_logits, gcn_logits, attn_weight