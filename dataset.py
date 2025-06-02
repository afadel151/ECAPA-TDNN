import os
import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as T



class LibriSpeechDataset(Dataset):
    def __init__(self, dataset, max_length=3.0, sample_rate=16000, n_mels=80):
        self.dataset = dataset
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, hop_length=160, win_length=400, n_fft=512
        )
        self.speaker_ids = sorted(set(item[2] for item in dataset))
        self.speaker_to_idx = {spk: idx for idx, spk in enumerate(self.speaker_ids)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sr, speaker_id, _, _ = self.dataset[idx]
        if sr != self.sample_rate:
            waveform = T.Resample(sr, self.sample_rate)(waveform)
        max_samples = int(self.max_length * self.sample_rate)
        if waveform.size(1) > max_samples:
            waveform = waveform[:, :max_samples]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, max_samples - waveform.size(1)))
        mel_spec = self.mel_transform(waveform).squeeze(0)
        mel_spec = torch.log(mel_spec + 1e-10)
        label = self.speaker_to_idx[speaker_id]
        return mel_spec, label
