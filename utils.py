import torch
import torch.nn as nn

class AAMSoftmax(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        super().__init__()
        self.s = s
        self.m = m
        self.weights = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weights)

    
    
def augment_audio(waveform, sample_rate):
    noise = torch.randn_like(waveform) * 0.005
    waveform = waveform + noise
    return waveform
