import torch
import torch.nn as nn

class AAMSoftmax(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        super().__init__()
        self.s = s
        self.m = m
        self.weights = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weights)
    def forward(self, embeddings, labels):
        cos_theta = torch.matmul(torch.nn.functional.normalize(embeddings), torch.nn.functional.normalize(self.weights, dim=0))
        cos_theta = torch.clamp(cos_theta, -1, 1)
        phi = cos_theta - self.m
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = self.s * (one_hot * phi + (1 - one_hot) * cos_theta)
        return torch.nn.functional.cross_entropy(output, labels)
    
    
def augment_audio(waveform, sample_rate):
    noise = torch.randn_like(waveform) * 0.005
    waveform = waveform + noise
    return waveform
