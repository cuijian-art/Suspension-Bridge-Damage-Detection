from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.nn import BatchNorm, ChebConv
from torch_geometric.utils import dropout_adj


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
        )
        self.projection = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

    def forward(self, inputs):
        features = self.layers(inputs)
        return self.projection(features.flatten(1))


class GraphGenerationLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(256, 10), nn.Sigmoid())

    def forward(self, features):
        attributes = self.embedding(features.flatten(1))
        adjacency = attributes @ attributes.transpose(0, 1)
        adjacency = adjacency / adjacency.max(dim=1, keepdim=True).values.clamp_min(1e-8)
        node_count = adjacency.size(0)
        source_nodes = torch.arange(node_count, device=features.device).repeat_interleave(node_count)
        target_nodes = torch.arange(node_count, device=features.device).repeat(node_count)
        edge_index = torch.stack((source_nodes, target_nodes), dim=0)
        return adjacency.reshape(-1), edge_index


class MultiReceptiveGraphBlock(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.branches = nn.ModuleList(
            ChebConv(input_features, output_features, K=order) for order in (1, 2, 3)
        )

    def forward(self, features, edge_index, edge_weight):
        return torch.cat(
            [branch(features, edge_index, edge_weight) for branch in self.branches],
            dim=1,
        )


class GraphFeatureExtractor(nn.Module):
    def __init__(self, edge_dropout=0.5):
        super().__init__()
        self.edge_dropout = float(edge_dropout)
        self.graph = GraphGenerationLayer()
        self.block1 = MultiReceptiveGraphBlock(256, 400)
        self.norm1 = BatchNorm(1200)
        self.block2 = MultiReceptiveGraphBlock(1200, 100)
        self.norm2 = BatchNorm(300)
        self.projection = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

    def forward(self, features):
        edge_weight, edge_index = self.graph(features)
        edge_index, edge_weight = dropout_adj(
            edge_index,
            edge_weight,
            p=self.edge_dropout,
            training=self.training,
        )
        features = self.norm1(self.block1(features, edge_index, edge_weight))
        features = self.norm2(self.block2(features, edge_index, edge_weight))
        return self.projection(features.flatten(1))


class DAGNNFeatureExtractor(nn.Module):
    def __init__(self, use_gcn=True, edge_dropout=0.5):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.gcn = GraphFeatureExtractor(edge_dropout) if use_gcn else nn.Identity()

    def forward(self, inputs):
        return self.gcn(self.cnn(inputs))


class DomainDiscriminator(nn.Module):
    def __init__(self, input_features=256, hidden_size=1024, max_iterations=10000):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self.iteration = 0
        self.max_iterations = max(float(max_iterations), 1.0)

    def forward(self, features):
        if self.training:
            self.iteration += 1
        coefficient = 2.0 / (1.0 + math.exp(-10.0 * self.iteration / self.max_iterations)) - 1.0
        reversed_features = features * 1.0
        if reversed_features.requires_grad:
            reversed_features.register_hook(lambda gradient: -coefficient * gradient.clone())
        return self.layers(reversed_features).flatten()
