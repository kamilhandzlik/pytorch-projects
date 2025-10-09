"""
Dataset used for training this model is from:
https://www.kaggle.com/datasets/abdulvahap/music-instrunment-sounds-for-classification?utm_source=chatgpt.com

can be downloaded using this code
import kagglehub

# Download latest version
path = kagglehub.dataset_download("abdulvahap/music-instrunment-sounds-for-classification")

print("Path to dataset files:", path)

or manually from link above it consist of 3 sec instrumental audioclips.

Ps. You have to have an account on kaggle to donwload this data.
PPs. Warning it weights aproximetly 5 Gb ;)
"""

# 1. Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as T
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys


# 2. Setting up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\33[91m------------------------------------\33[0m")
print("\33[91m|    Sound classification model     |\33[0m")
print("\33[91m------------------------------------\33[0m")
print(
    f"\n\033[32mDevice is set to: {device}\033[0m\n"
    if device == "cuda"
    else f"\n\033[34mDevice is set to: {device}\033[0m\n"
)
# torchaudio.set_audio_backend("soundfile")
# print(f"torchaudio: {torchaudio}")


# 3. Hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
INPUT_SHAPE = "?"
HIDDEN_UNITS = 128
OUTPUT = "?"
LOSS_FN = nn.CrossEntropyLoss()
N_ROWS = 3
N_COLS = 3


# 4. Reading dataset
sound_path = []
labels = []

for i in os.listdir("Sound_Classification/dataset/"):
    for label in os.listdir(f"Sound_Classification/dataset/{i}"):
        for sound in os.listdir(f"Sound_Classification/dataset/{i}/{label}"):
            labels.append(label)
            sound_path.append(f"Sound_Classification/dataset/{i}/{label}/{sound}")


data_df = pd.DataFrame(zip(sound_path, labels), columns=["sound_path", "labels"])
print(data_df.head())


# 5. Data split
train = data_df.sample(frac=0.7, random_state=RANDOM_SEED)
temp = data_df.drop(train.index)
val = temp.sample(frac=0.5, random_state=RANDOM_SEED)
test = temp.drop(val.index)

print(f"train: {train}")
print(f"test: {temp}")
print(f"val: {val}")
print(f"test_val: {test}")


# 6. Preprocessing Objects
label_encoder = LabelEncoder()
label_encoder.fit(data_df["labels"])

# Mel-spectogram
transform = nn.Sequential(
    T.MelSpectrogram(sample_rate=44100, n_fft=1024, hop_length=128), T.AmplitudeToDB()
)


# 7. Custom Dataset Class
class InstrumentDataset(Dataset):
    def __init__(self, df, label_encoder, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sound_path = self.df.loc[idx, "sound_path"]
        label = self.df.loc[idx, "labels"]

        waveform, sample_rate = torchaudio.load(sound_path)
        label = self.label_encoder.transform([label])[0]

        if self.transform:
            features = self.transform(waveform)
        else:
            features = waveform

        return features, label


# 8. Create Dataset Objects
train_dataset = InstrumentDataset(train, label_encoder, transform)
temp_dataset = InstrumentDataset(temp, label_encoder, transform)
val_dataset = InstrumentDataset(val, label_encoder, transform)
test_dataset = InstrumentDataset(test, label_encoder, transform)


# 9. Vizualization of data
def plot_waveform(waveform, sample_rate, title="Waveform"):
    plt.figure(figsize=(10, 4))
    plt.plot(
        np.linspace(0, waveform.shape[1] / sample_rate, waveform.shape[1]),
        waveform[0].numpy(),
    )
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()


def plot_spectrogram(spec, title="Spectrogram"):
    plt.figure(figsize=(10, 4))
    plt.imshow(spec[0].numpy(), aspect="auto", origin="lower")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Mel bins")
    plt.colorbar(format="%+2.0f dB")
    plt.show()


def visualize_batch(dataloader, n_rows=3, n_cols=3, mode="spectrogram"):
    batch = next(iter(dataloader))
    features, labels = batch

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= len(features):
            break

        if mode == "spectrogram":
            spec = features[i][0].numpy()
            ax.imshow(spec, aspect="auto", origin="lower")
            ax.set_title(f"Label: {labels[i].item()}")
        else:
            waveform = features[i][0].numpy()
            ax.plot(waveform)
            ax.set_title(f"Label: {labels[i].item()}")

        ax.axis("off")

    plt.tight_layout()
    plt.show()


train_loader = DataLoader(train_dataset, batch_size=9, shuffle=True)

visualize_batch(train_loader, mode="spectrogram")


# 10. Dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# 11. CNN Model


class CNNSoundClassifeir(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Con blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# 14. Train and test loops
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 180
    return acc


def train_step(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn=None,
    device: torch.device = device,
):
    train_loss, train_loss = 0, 0
    model.to(device)
    model.train

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        # 1. Forward
        y_pred = model(X)

        # 2. calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Calculate accuracy
        if accuracy_fn:
            train_acc = accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # 4. Optmimizer zero grad
        optimizer.zerograd()

        # 5. Backward
        loss.backward()

        # 6. Optmizer step
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    print(f"Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f}%")
    return train_loss, train_acc


def test_step(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    accuracy_fn,
    device: torch.device = device,
):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            # Forward pass
            test_pred = model(X)

            # Calculate the loss
            test_loss += loss_fn(test_pred, y).item()
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        test_loss += len(data_loader)
        test_acc += len(data_loader)

    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}%")
    return test_loss, test_acc
