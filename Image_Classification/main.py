# 1. Initializations and Dataset Download

# import kagglehub

# Download latest version
# path = kagglehub.dataset_download("marquis03/bean-leaf-lesions-classification")

# print("Path to dataset files:", path)


# You can download manually from https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification


# 2. Imports
import torch
from torch import nn
from torch.optim import Adam, SGD
from torchvision import models
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
import sys

# 3. Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\33[91m------------------------------------\33[0m")
print(f"\n\33[92mUsing device: {device}\33[0m\n")


# 4. Hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
N_ROWS = 3
N_COLS = 3

# 5. Reading data paths
train_df = pd.read_csv("Image_Classification/dataset/train.csv")
val_df = pd.read_csv("Image_Classification/dataset/val.csv")

data_df = pd.concat([train_df, val_df], ignore_index=True)
data_df["image:FILE"] = "Image_Classification/dataset/" + data_df["image:FILE"]

print(f"data_df.head(): {data_df.head()}")
print(f"data_df.shape: {data_df.shape}")


# 6. Inspecting data

print("\n\nClasses are: ")
print(data_df["category"].unique())
print(f"\n\nClasses distribution are: \n{data_df['category'].value_counts()}")


# 7. Splitting data
train = data_df.sample(frac=0.8, random_state=RANDOM_SEED)
test = data_df.drop(train.index)


# 8. Preprocessing of data / objects in data ig resizing images and turning to tensors
label_encoder = LabelEncoder()
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
    ]
)

# 9. Custom Dataset Class


class CustomDatasetClass(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = torch.tensor(
            label_encoder.fit_transform(dataframe["category"])
        ).to(device)

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        image_path = self.dataframe.iloc[index, 0]
        label = self.labels[index]
        image = Image.open(image_path).convert("RGB")
        if transform:
            image = (self.transform(image) / 255).to(device)

        return image, label


# 10. Datasets

train_dataset = CustomDatasetClass(train, transform)
test_dataset = CustomDatasetClass(test, transform)


# 11. Visualize data


def visualize_data(dataset, num_images=5):
    f, axarr = plt.subplots(N_ROWS, N_COLS)
    for row in range(N_ROWS):
        for col in range(N_COLS):
            image = train_dataset[np.random.randint(0, train_dataset.__len__())][
                0
            ].cpu()
            axarr[row, col].imshow((image * 255).squeeze().permute(1, 2, 0))
            axarr[row, col].axis("off")
    plt.tight_layout()
    plt.show()


visualize_data(train_dataset)


# 12. Dataloaders


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# 13. Model


class ImageClassificationModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# 14 Train and test loops
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def train_step(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn=None,
    device: torch.device = device,
):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in data_loader:
        X, y = X.to(device), y.to(device)
        model.train()
        # 1. Forward pass
        y_pred = model(X)
        # 2. Calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        # 3. Optimizer zerograd
        optimizer.zero_grad()
        # 4.  Backward pass
        loss.backward()
        # 5. Optmizer step
        optimizer.step()
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss} | Train accuracy: {train_acc}%")


def test_step(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn,
    device: torch.device = device,
):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            # 1. Forward pass
            test_pred = model(X)
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
    print(f"Test loss: {test_loss} | Test accuracy: {test_acc}%")


for epoch in range(EPOCHS):
    train_step()
    test_step()
