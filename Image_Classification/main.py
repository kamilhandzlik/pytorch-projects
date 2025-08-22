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

# 3. Hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10

# 4. Reading data paths
train_df = pd.read_csv("Image_Classification/dataset/train.csv")
val_df = pd.read_csv("Image_Classification/dataset/val.csv")

data_df = pd.concat([train_df, val_df], ignore_index=True)
data_df["image:FILE"] = "Image_Classification/dataset/" + data_df["image:FILE"]

print(f"data_df.head(): {data_df.head()}")
print(f"data_df.shape: {data_df.shape}")


# 5. Inspecting data

print("\n\nClasses are: ")
print(data_df["category"].unique())
print(f"\n\nClasses distribution are: \n{data_df['category'].value_counts()}")


# 6. Splitting data
train = data_df.sample(frac=0.8, random_state=RANDOM_SEED)
test = data_df.drop(train.index)
