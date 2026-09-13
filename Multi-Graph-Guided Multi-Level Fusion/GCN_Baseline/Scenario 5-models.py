import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# 论文3.2.3节的CAM通道注意力模块
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
        # x: [B, C, L]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class MultiLevelFusionModel(nn.Module):
    def __init__(self, n_sensor=6, num_classes=6, window_size=1024):
        super().__init__()
        self.n_sensor = n_sensor
        self.num_classes = num_classes
        # ---------------------- 对齐论文Table2的CNN参数 ----------------------
        self.cnn = nn.Sequential(
            # 第一层
            nn.Conv1d(1, 32, kernel_size=25, stride=1, padding=12),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            # 第二层
            nn.Conv1d(32, 24, kernel_size=13, stride=1, padding=6),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            # 第三层
            nn.Conv1d(24, 16, kernel_size=13, stride=1, padding=6),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.AdaptiveAvgPool1d(1)
        )
        self.cnn_out_dim = 16
        self.cam = CAM(in_channels=self.cnn_out_dim)

        # ---------------------- 对齐论文Table3的GCN参数 ----------------------
        self.gcn_conv1 = GCNConv(self.cnn_out_dim, 256)
        self.gcn_conv2 = GCNConv(256, 64)
        self.gcn_conv3 = GCNConv(64, 16)
        self.gcn_out_dim = 16

        # ---------------------- 论文创新点2：注意力特征融合模块 ----------------------
        self.attn_fc = nn.Sequential(
            nn.Linear(self.cnn_out_dim + self.gcn_out_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )
        self.attn_softmax = nn.Softmax(dim=1)

        # 单分支分类头，用于辅助损失和后续决策融合
        self.cnn_cls = nn.Linear(self.cnn_out_dim, num_classes)
        self.gcn_cls = nn.Linear(self.gcn_out_dim, num_classes)

    def forward(self, x, edge_index, edge_weight, n_per_sensor):
        batch_size = x.shape[0]
        # ---------------------- 1. 单传感器CNN特征提取 ----------------------
        cnn_feat_list = []
        cnn_logits_list = []
        for s in range(self.n_sensor):
            s_x = x[:, s:s+1, :] # [B, 1, 1024]
            s_feat = self.cnn(s_x).squeeze(-1) # [B, 16]
            # 加CAM通道增强
            s_feat_cam = self.cam(s_feat.unsqueeze(-1)).squeeze(-1)
            cnn_feat_list.append(s_feat_cam)
            cnn_logits_list.append(self.cnn_cls(s_feat_cam))
        cnn_feat = torch.stack(cnn_feat_list, dim=1) # [B, 6, 16]
        cnn_logits = torch.stack(cnn_logits_list, dim=1) # [B, 6, num_classes]

        # ---------------------- 2. GCN特征提取 ----------------------
        gcn_feat = cnn_feat.reshape(-1, self.cnn_out_dim) # [B*6, 16]
        gcn_feat = F.relu(self.gcn_conv1(gcn_feat, edge_index, edge_weight))
        gcn_feat = F.relu(self.gcn_conv2(gcn_feat, edge_index, edge_weight))
        gcn_feat = self.gcn_conv3(gcn_feat, edge_index, edge_weight)
        gcn_feat = gcn_feat.reshape(batch_size, self.n_sensor, self.gcn_out_dim) # [B, 6, 16]
        gcn_logits = self.gcn_cls(gcn_feat) # [B, 6, num_classes]

        # ---------------------- 论文创新点2：双向注意力融合 ----------------------
        concat_feat = torch.cat([cnn_feat, gcn_feat], dim=-1) # [B, 6, 32]
        attn_score = self.attn_fc(concat_feat) # [B, 6, 1]
        attn_weight = self.attn_softmax(attn_score) # [B, 6, 1]
        # 加权融合特征
        fused_feat = (attn_weight * concat_feat).flatten(1) # [B, 6*32=192]

        # ---------------------- 输出所有分支结果，用于后续决策融合 ----------------------
        return fused_feat, cnn_logits, gcn_logits, attn_weight