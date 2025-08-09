# 1. Imports
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
import sys

# 2. Setting up the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 3 Hyperparameters
RANDOM_SEED = 42
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 10

# 4. Dataset
image_path = []
labels = []

for i in os.listdir("Animal_Image_Classification/afhq"):
    for label in os.listdir(f"Animal_Image_Classification/afhq/{i}"):
        for image in os.listdir(f"Animal_Image_Classification/afhq/{i}/{label}"):
            labels.append(label)
            image_path.append(f"Animal_Image_Classification/afhq/{i}/{label}/{image}")


data_df = pd.DataFrame(zip(image_path, labels), columns=["image_paths", "labels"])
print(data_df.head())


# 5. Data splitting
train = data_df.sample(frac=0.8, random_state=RANDOM_SEED)
test = data_df.drop(train.index)

val = test.sample(frac=0.5, random_state=RANDOM_SEED)
test = test.drop(val.index)


# 6. Preprocessing Objects
label_encoder = LabelEncoder()
label_encoder.fit(data_df["labels"])

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
    ]
)

# 7. Custom Dataset Class


class AnimalDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = torch.tensor(label_encoder.transform(dataframe["labels"])).to(
            device
        )

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image).to(device)

        return image, label


# 8. Create Dataset Objects
train_dataset = AnimalDataset(train, transform)
val_dataset = AnimalDataset(val, transform)
test_dataset = AnimalDataset(test, transform)


# 9. Vizualization of data
n_rows = 3
n_cols = 3
f, axarr = plt.subplots(n_rows, n_cols)
for row in range(n_rows):
    for col in range(n_cols):
        image = Image.open(data_df.sample(n=1)["image_paths"].iloc[0]).convert("RGB")
        axarr[row, col].imshow(image)
        axarr[row, col].axis("off")

plt.show()  # Comment if you don't want to see the images


# 10. DataLoaders

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# 11. Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.linear = nn.Linear((128 * 16 * 16), 128)
        self.output = nn.Linear(128, len(data_df["labels"].unique()))

    def forward(self, x):
        x = self.conv1(x)  # -> Outputs: (32, 128, 128)
        x = self.pooling(x)  # -> Outputs: (32, 64, 64)
        x = self.relu(x)
        x = self.conv2(x)  # -> Outputs: (64, 64, 64)
        x = self.pooling(x)  # -> Outputs: (64, 32, 32)
        x = self.relu(x)
        x = self.conv3(x)  # -> Outputs: (128, 32, 32)
        x = self.pooling(x)  # -> Outputs: (128, 16, 16)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.output(x)

        return x


model = Net().to(device)
