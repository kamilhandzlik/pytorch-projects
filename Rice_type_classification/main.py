# import opendatasets as od
# od.download("https://www.kaggle.com/datasets/mssmartypants/rice-type-classification")


import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


csv_path = os.path.join(os.path.dirname(__file__), "riceClassification.csv")
data_df = pd.read_csv(csv_path)
data_df.head()
