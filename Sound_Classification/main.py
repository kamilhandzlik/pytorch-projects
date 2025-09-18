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
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
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
