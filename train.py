from dataset import download_librispeech,LibriSpeechDataset
import torch
from torch.utils.data import DataLoader
from model import ECAPATDNN,train_epoch,evaluate
from utils import AAMSoftmax
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset, test_dataset = download_librispeech()

n_train = int(0.8 * len(train_dataset))
n_val = len(train_dataset) - n_train
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [n_train, n_val])

train_loader = DataLoader(LibriSpeechDataset(train_subset), batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(LibriSpeechDataset(val_subset), batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(LibriSpeechDataset(test_dataset), batch_size=32, shuffle=False, num_workers=4)

model = ECAPATDNN().to(device)

criterion = AAMSoftmax(192, 251).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_eer = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, EER: {val_eer:.4f}")
    scheduler.step()
    
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'train_loss': train_loss,
    'val_eer': val_eer
}, 'ecapa_tdnn_checkpoint.pth')