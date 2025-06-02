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


class TDNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation // 2, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        return x

class ECAPATDNN(nn.Module):
    def __init__(self, in_channels=80, channels=512, embedding_size=192, num_classes=251):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, channels, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.tdnn_blocks = nn.ModuleList([
            TDNNBlock(channels, channels, 3, dilation=2),
            TDNNBlock(channels, channels, 3, dilation=3),
            TDNNBlock(channels, channels, 3, dilation=4)
        ])
        self.agg_conv = nn.Conv1d(channels * 3, 1536, 1)
        self.pooling = AttentiveStatPooling(1536)
        self.fc1 = nn.Linear(1536 * 2, embedding_size)
        self.fc2 = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        tdnn_outputs = []
        for block in self.tdnn_blocks:
            x = block(x)
            tdnn_outputs.append(x)
        x = torch.cat(tdnn_outputs, dim=1)
        x = self.agg_conv(x)
        x = self.pooling(x)
        embedding = self.fc1(x)
        x = self.fc2(embedding)
        return embedding, x

def compute_eer(scores, labels):
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch_idx, (data, labels) in enumerate(loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings, outputs = model(data)
        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return total_loss / len(loader), 100 * correct / total


def evaluate(model, loader, device):
    model.eval()
    embeddings_list, labels_list = [], []
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            embeddings, _ = model(data)
            embeddings_list.append(embeddings.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    embeddings = np.concatenate(embeddings_list)
    labels = np.concatenate(labels_list)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_emb = tsne.fit_transform(embeddings)
    plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=labels, cmap='viridis')
    plt.savefig('tsne.png')
    # Compute EER (simplified pairwise cosine similarity)
    scores, true_labels = [], []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            score = torch.cosine_similarity(
                torch.tensor(embeddings[i:i+1]), torch.tensor(embeddings[j:j+1])
            ).item()
            scores.append(score)
            true_labels.append(1 if labels[i] == labels[j] else 0)
    eer = compute_eer(scores, true_labels)
    return eer


def test_model(model, loader,criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            _, outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.savefig('confusion_matrix.png')
    eer = evaluate(model, loader, criterion, device)
    print(f"Test EER: {eer:.4f}")
