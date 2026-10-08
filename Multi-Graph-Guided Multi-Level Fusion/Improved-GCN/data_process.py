import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from scipy.fft import fft
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

# ====================== 你原来的配置区，不需要改 ======================
ROOT_PATH = r"F:\A数据集\A悬索桥"
CLASS_LIST = ["C1", "C2", "C3", "C4", "C5", "C6"]
N_SENSOR = 6
WINDOW_SIZE = 1024
STRIDE = 1024
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
CACHE_PATH = "./data_cache_multi_graph.npz"
USE_CACHE = True
# 新增：多重图超参，可按需调整
SIMILARITY_RATIO = 0.5
KNN_K = 7
COS_QUANTILE = 0.7
USE_MST = True  # 开最小生成树保证图无孤立节点
USE_MD = True   # 开马氏距离提升抗噪性，建图慢可以关
# ======================================================

# ====================== 新增：多重图构建模块 ======================
class GraphConstructor:
    def __init__(self, knn_k=5, cos_quantile=0.75, device='cpu',
                 use_mst=False, use_md=False, reduce_dim=256):
        self.knn_k = knn_k
        self.cos_quantile = cos_quantile
        self.device = device
        self.use_mst = use_mst
        self.use_md = use_md
        self.reduce_dim = reduce_dim
        self.random_proj = None

    def _random_project(self, X):
        B, D = X.shape
        if self.reduce_dim <= 0 or D <= self.reduce_dim:
            return X
        if self.random_proj is None:
            self.random_proj = torch.randn(D, self.reduce_dim, device=self.device) / (self.reduce_dim ** 0.5)
        return X @ self.random_proj

    def build_knn_graph(self, X):
        B, D = X.shape
        dist = torch.cdist(X, X)
        _, nn_idx = torch.topk(dist, k=self.knn_k + 1, largest=False, dim=-1)
        adj = torch.zeros(B, B, device=self.device)
        for i in range(B):
            adj[i, nn_idx[i, 1:]] = 1
        adj = adj + adj.T
        adj[adj > 1] = 1
        adj.fill_diagonal_(0)
        return adj

    def build_cos_graph(self, X):
        B, D = X.shape
        X_norm = torch.nn.functional.normalize(X, p=2, dim=-1)
        cos_sim = X_norm @ X_norm.T
        # threshold = torch.quantile(cos_sim.flatten(), self.cos_quantile)
        threshold = np.quantile(cos_sim.flatten().numpy(), self.cos_quantile)
        adj = (cos_sim >= threshold).float()
        adj.fill_diagonal_(0)
        adj = adj + adj.T
        adj[adj > 1] = 1
        return adj

    def build_mst_graph(self, X):
        B, D = X.shape
        X_np = X.detach().cpu().numpy()
        dist = squareform(pdist(X_np, metric='euclidean'))
        mst = minimum_spanning_tree(dist)
        adj = torch.tensor(mst.toarray(), dtype=torch.float32, device=self.device)
        adj = adj + adj.T
        adj[adj > 1] = 1
        adj.fill_diagonal_(0)
        return adj

    def build_md_graph(self, X):
        B, D = X.shape
        if D > 1024:
            return torch.zeros(B, B, device=self.device)
        X_np = X.detach().cpu().numpy()
        cov = np.cov(X_np.T) + 1e-3 * np.eye(D)
        inv_cov = np.linalg.inv(cov)
        dist = squareform(pdist(X_np, metric='mahalanobis', VI=inv_cov))
        threshold = np.quantile(dist.flatten(), 0.25)
        adj_np = (dist <= threshold).astype(np.float32)
        adj = torch.tensor(adj_np, dtype=torch.float32, device=self.device)
        adj.fill_diagonal_(0)
        adj = adj + adj.T
        adj[adj > 1] = 1
        return adj

    def fuse_graphs(self, adj_list):
        adj_list = [a for a in adj_list if a.sum() > 0]
        num_graphs = len(adj_list)
        if num_graphs == 1:
            return adj_list[0]
        W = torch.zeros(num_graphs, num_graphs, device=self.device)
        for i in range(num_graphs):
            for j in range(num_graphs):
                W[i, j] = torch.exp(-torch.norm(adj_list[i] - adj_list[j], p='fro'))
        w = W.sum(dim=1) / W.sum()
        fused_adj = torch.zeros_like(adj_list[0])
        for i in range(num_graphs):
            fused_adj += w[i] * adj_list[i]
        median_val = torch.median(fused_adj.flatten())
        fused_bin = (fused_adj > median_val).float()
        return fused_bin

    def __call__(self, X):
        X_low = self._random_project(X)
        adj_list = [self.build_knn_graph(X_low), self.build_cos_graph(X_low)]
        if self.use_mst:
            adj_list.append(self.build_mst_graph(X_low))
        if self.use_md:
            adj_list.append(self.build_md_graph(X_low))
        return self.fuse_graphs(adj_list)

# ====================== 原有加载逻辑，仅替换建图部分 ======================
def load_data():
    if USE_CACHE and os.path.exists(CACHE_PATH):
        data = np.load(CACHE_PATH)
        x = torch.FloatTensor(data['x'])
        y = torch.LongTensor(data['y'])
        edge_index = torch.LongTensor(data['edge_index'])
        edge_weight = torch.FloatTensor(data['edge_weight'])
        train_mask = torch.BoolTensor(data['train_mask'])
        val_mask = torch.BoolTensor(data['val_mask'])
        test_mask = torch.BoolTensor(data['test_mask'])
        return x, y, edge_index, edge_weight, train_mask, val_mask, test_mask, x.shape[0], N_SENSOR, len(CLASS_LIST)

    all_x = []
    all_y = []
    all_fft = []
    for cls_idx, cls_name in enumerate(CLASS_LIST):
        cls_path = os.path.join(ROOT_PATH, cls_name)
        sensor_data = []
        min_len = float('inf')
        for s in range(1, N_SENSOR+1):
            file_path = os.path.join(cls_path, f"AI1-{s:02d}.xlsx")
            arr = pd.read_excel(file_path, header=None).iloc[:200000].values.squeeze()
            sensor_data.append(arr)
            if len(arr) < min_len:
                min_len = len(arr)
        # 滑窗切样本
        num_sample = (min_len - WINDOW_SIZE) // STRIDE + 1
        for i in range(num_sample):
            start = i * STRIDE
            end = start + WINDOW_SIZE
            sample = np.stack([s_data[start:end] for s_data in sensor_data], axis=0)
            sample = (sample - sample.mean(axis=1, keepdims=True)) / (sample.std(axis=1, keepdims=True) + 1e-8)
            all_x.append(sample)
            all_y.append(cls_idx)
            # 计算FFT特征
            sample_fft = np.abs(fft(sample, axis=1))[:, :WINDOW_SIZE//2]
            all_fft.append(sample_fft)

    all_x = np.stack(all_x, axis=0)
    all_y = np.array(all_y)
    all_fft = np.stack(all_fft, axis=0)
    total_sample = len(all_x)

    # ====================== 替换原有建图逻辑为多重图 ======================
    print(f"正在构建多重图，总节点数：{total_sample * N_SENSOR}...")
    graph_builder = GraphConstructor(knn_k=KNN_K, cos_quantile=COS_QUANTILE, use_mst=USE_MST, use_md=USE_MD)
    # 每个节点对应一个传感器的FFT特征：形状[总节点数, 512]
    node_fft = torch.FloatTensor(all_fft.reshape(-1, all_fft.shape[-1]))
    adj = graph_builder(node_fft)
    # 保留原有强先验：同一个样本的不同传感器强制连边
    for i in range(total_sample):
        for s1 in range(N_SENSOR):
            for s2 in range(s1+1, N_SENSOR):
                u = i * N_SENSOR + s1
                v = i * N_SENSOR + s2
                adj[u, v] = 1.0
                adj[v, u] = 1.0
    # 稠密邻接矩阵转稀疏edge_index，和原有格式完全兼容
    edge_index = adj.nonzero().T.long()
    edge_weight = adj[edge_index[0], edge_index[1]].float()
    # ======================================================

    # 原有数据集划分逻辑完全保留
    train_idx, temp_idx = train_test_split(np.arange(total_sample), test_size=VAL_RATIO+TEST_RATIO, stratify=all_y, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO), stratify=all_y[temp_idx], random_state=42)
    train_mask = np.zeros(total_sample, dtype=bool)
    train_mask[train_idx] = True
    val_mask = np.zeros(total_sample, dtype=bool)
    val_mask[val_idx] = True
    test_mask = np.zeros(total_sample, dtype=bool)
    test_mask[test_idx] = True

    # 存缓存
    np.savez(CACHE_PATH, x=all_x, y=all_y, edge_index=edge_index.numpy(), edge_weight=edge_weight.numpy(),
             train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return torch.FloatTensor(all_x), torch.LongTensor(all_y), edge_index, edge_weight, \
           torch.BoolTensor(train_mask), torch.BoolTensor(val_mask), torch.BoolTensor(test_mask), total_sample, N_SENSOR, len(CLASS_LIST)