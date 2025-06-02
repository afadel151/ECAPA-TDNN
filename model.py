import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class AttentiveStatPooling(nn.Module):
    def __init__(self, channels, attention_channels=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_channels, 1), nn.ReLU(),
            nn.BatchNorm1d(attention_channels), nn.Conv1d(attention_channels, channels, 1), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        weights = self.attention(x)
        mean = torch.sum(x * weights, dim=-1)
        std = torch.sqrt(torch.sum((x ** 2) * weights, dim=-1) - mean ** 2 + 1e-10)
        return torch.cat([mean, std], dim=-1)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(), nn.Conv1d(channels // reduction, channels, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)
