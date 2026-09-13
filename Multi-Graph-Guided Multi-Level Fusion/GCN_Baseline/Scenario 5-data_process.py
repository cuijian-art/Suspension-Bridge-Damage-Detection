import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from scipy.fft import fft
from sklearn.metrics.pairwise import cosine_similarity

# ====================== 配置区 ======================
ROOT_PATH = r"F:\A数据集\A悬索桥"
CLASS_LIST = ["C1", "C2", "C3", "C4", "C5", "C6"]
N_SENSOR = 6
WINDOW_SIZE = 1024
STRIDE = 1024
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
CACHE_PATH = "./S2_data_cache_multi_graph.npz"
USE_CACHE = True
# 论文自适应边阈值超参，可调
SIMILARITY_RATIO = 0.5  # 保留前50%相似度的边，对应论文公式9的自适应逻辑


# ======================================================

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
    all_fft = []  # 存储每个样本的FFT频谱，用于计算边相似度
    for cls_idx, cls_name in enumerate(CLASS_LIST):
        cls_path = os.path.join(ROOT_PATH, cls_name)
        sensor_data = []
        min_len = float('inf')
        for s in range(1, N_SENSOR + 1):
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
            # 计算每个样本各传感器的FFT频谱（对应论文公式8的F(x)）
            sample_fft = np.abs(fft(sample, axis=1))[:, :WINDOW_SIZE // 2]
            all_fft.append(sample_fft)

    all_x = np.stack(all_x, axis=0)  # [总样本数, 6, 1024]
    all_y = np.array(all_y)
    all_fft = np.stack(all_fft, axis=0)  # [总样本数, 6, 512]
    total_sample = len(all_x)

    # ====================== 论文创新点1：多层动态图构建 ======================
    edge_index = []
    edge_weight = []
    # 1. 单传感器内部边：样本间FFT余弦相似度大于阈值才连通
    for s in range(N_SENSOR):
        s_fft = all_fft[:, s, :]  # 当前传感器所有样本的FFT
        sim_matrix = cosine_similarity(s_fft)  # 两两样本相似度
        # 取前SIMILARITY_RATIO的高相似度边
        threshold = np.quantile(sim_matrix, 1 - SIMILARITY_RATIO)
        for i in range(total_sample):
            for j in range(i + 1, total_sample):
                if sim_matrix[i, j] >= threshold:
                    # 节点索引规则：第i个样本第s个传感器的节点id = i*N_SENSOR + s
                    u = i * N_SENSOR + s
                    v = j * N_SENSOR + s
                    edge_index.append([u, v])
                    edge_index.append([v, u])
                    edge_weight.append(sim_matrix[i, j])
                    edge_weight.append(sim_matrix[i, j])
    # 2. 跨传感器边：同一个样本的不同传感器强制连通（对应同一时刻的设备状态）
    for i in range(total_sample):
        for s1 in range(N_SENSOR):
            for s2 in range(s1 + 1, N_SENSOR):
                u = i * N_SENSOR + s1
                v = i * N_SENSOR + s2
                edge_index.append([u, v])
                edge_index.append([v, u])
                edge_weight.append(1.0)  # 同一时刻跨传感器边权固定为1
                edge_weight.append(1.0)

    edge_index = np.array(edge_index).T
    edge_weight = np.array(edge_weight, dtype=np.float32)

    # 分层划分数据集
    train_idx, temp_idx = train_test_split(np.arange(total_sample), test_size=VAL_RATIO + TEST_RATIO, stratify=all_y,
                                           random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
                                         stratify=all_y[temp_idx], random_state=42)
    train_mask = np.zeros(total_sample, dtype=bool)
    train_mask[train_idx] = True
    val_mask = np.zeros(total_sample, dtype=bool)
    val_mask[val_idx] = True
    test_mask = np.zeros(total_sample, dtype=bool)
    test_mask[test_idx] = True

    # 存缓存
    np.savez(CACHE_PATH, x=all_x, y=all_y, edge_index=edge_index, edge_weight=edge_weight,
             train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return torch.FloatTensor(all_x), torch.LongTensor(all_y), torch.LongTensor(edge_index), torch.FloatTensor(
        edge_weight), \
           torch.BoolTensor(train_mask), torch.BoolTensor(val_mask), torch.BoolTensor(
        test_mask), total_sample, N_SENSOR, len(CLASS_LIST)