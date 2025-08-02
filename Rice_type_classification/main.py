# 0. Initializations and Dataset Download
# import opendatasets as od
# od.download("https://www.kaggle.com/datasets/mssmartypants/rice-type-classification")

# Imports
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

# 1. Setting up the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# 2. Dataset
csv_path = os.path.join(os.path.dirname(__file__), "riceClassification.csv")
data_df = pd.read_csv(csv_path)
# print(data_df.head())
data_df.dropna(inplace=True)
data_df.drop(["id"], axis=1, inplace=True)
# print(data_df.shape)
# print("Output possibilities: ", data_df["Class"].unique())
# print("Data shape (rows, cols): ", data_df.shape)
# print(data_df.head())

# 3. Data Preprocessing
original_df = (
    data_df.copy()
)  # Creating a copy of the original Dataframe to use to normalize inference

for column in data_df.columns:
    data_df[column] = data_df[column] / data_df[column].abs().max()

print(data_df.head())


# 4. Data Splitting
X = np.array(
    data_df.iloc[:, :-1]
)  # Get the inputs, all rows and all columns except last column (output)
Y = np.array(
    data_df.iloc[:, -1]
)  # Get the ouputs, all rows and last column only (output column)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3
)  # Create the training split
X_test, X_val, Y_test, Y_val = train_test_split(
    X_test, Y_test, test_size=0.5
)  # Create the validation split


# print(
# "Training set is: ",
# X_train.shape[0],
# "rows which is",
# round(X_train.shape[0] / data_df.shape[0], 4) * 100,
# "%",
# )
# print(
# "Validation set is: ",
# X_val.shape[0],
# "rows which is",
# round(X_val.shape[0] / data_df.shape[0], 4) * 100,
# "%",
# )
# print(
# "Testing set is: ",
# X_test.shape[0],
# "rows which is",
# round(X_test.shape[0] / data_df.shape[0], 4) * 100,
# "%",
# )


# 5. Dataset Object
class dataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.Y = torch.tensor(Y, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


training_data = dataset(X_train, Y_train)
validation_data = dataset(X_val, Y_val)
testing_data = dataset(X_test, Y_test)

# 6. Training Hyper Parameters
BATCH_SIZE = 32
EPOCHS = 10
HIDDEN_UNITS = 10
LEARNING_RATE = 1e-3

# 7. Data Loaders
training_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)
testing_dataloader = DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=True)

# 8. Model Class


class RiceClassifcationModel(nn.Module):
    def __init__(self):
        super(RiceClassifcationModel, self).__init__()

        self.input_layser = nn.Linear(X.shape[1], HIDDEN_UNITS)
        self.linear = nn.Linear(HIDDEN_UNITS, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layser(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


# 9. Model Initialization
model = RiceClassifcationModel().to(device)
# print(summary(model, (X.shape[1],)))


# 10. Loss Function and Optimizer
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)


# 11. Training Loop
total_loss_train_plot = []
total_loss_validation_plot = []
total_acc_train_plot = []
total_acc_validation_plot = []

for epoch in range(EPOCHS):
    total_acc_train = 0
    total_loss_train = 0
    total_acc_val = 0
    total_loss_val = 0

    for data in training_dataloader:
        inputs, labels = data
        predictions = model(inputs).squeeze()
        batch_loss = criterion(predictions, labels)
        total_loss_train += batch_loss.item()
        acc = ((predictions).round() == labels).sum().item()
        total_acc_train += acc

        batch_loss.backward()
        optimizer.zero_grad()
        optimizer.step()

    with torch.no_grad():
        for data in validation_dataloader:
            inputs, labels = data
            predictions = model(inputs).squeeze()
            batch_loss = criterion(predictions, labels)
            total_loss_val += batch_loss.item()
            acc = ((predictions).round() == labels).sum().item()
            total_acc_val += acc

    total_loss_train_plot.append(round(total_loss_train / 1000, 4))
    total_acc_train_plot.append(round(total_acc_train / len(training_data), 4))
    total_loss_validation_plot.append(round(total_loss_val / 1000, 4))
    total_acc_validation_plot.append(round(total_acc_val / len(validation_data), 4))

    print(
        f"""Epoch no. {epoch + 1} Train Loss: {total_loss_train/1000:.4f} Train Accuracy: {(total_acc_train/(training_data.__len__())*100):.4f} Validation Loss: {total_loss_val/1000:.4f} Validation Accuracy: {(total_acc_val/(validation_data.__len__())*100):.4f}"""
    )
    print("=" * 50)


# 12. Testing the Model

with torch.no_grad():
    total_acc_test = 0
    total_loss_test = 0

    for data in testing_dataloader:
        inputs, labels = data
        predictions = model(inputs).squeeze()
        batch_loss_test = criterion(predictions, labels)
        total_loss_test += batch_loss_test.item()
        acc = ((predictions).round() == labels).sum().item()
        total_acc_test += acc

print(f"Accuracy Score is: {round(total_acc_test / X_test.shape[0])*100, 2}%")

# 13. Plotting and Visualization

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

axs[0].plot(total_loss_train_plot, label="Training Loss")
axs[0].plot(total_loss_validation_plot, label="Validation Loss")
axs[0].set_title("Training and Validation Loss over Epochs")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[0].set_ylim([0, 2])
axs[0].legend()

axs[1].plot(total_acc_train_plot, label="Training Accuracy")
axs[1].plot(total_acc_validation_plot, label="Validation Accuracy")
axs[1].set_title("Training and Validation Accuracy over Epochs")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Accuracy")
axs[1].set_ylim([0, 100])
axs[1].legend()

plt.tight_layout()

plt.show()


# 14. Inference
area = float(input("Area: ")) / original_df["Area"].abs().max()
MajorAxisLength = (
    float(input("Major Axis Length: ")) / original_df["MajorAxisLength"].abs().max()
)
MinorAxisLength = (
    float(input("Minor Axis Length: ")) / original_df["MinorAxisLength"].abs().max()
)
Eccentricity = float(input("Eccentricity: ")) / original_df["Eccentricity"].abs().max()
ConvexArea = float(input("Convex Area: ")) / original_df["ConvexArea"].abs().max()
EquivDiameter = (
    float(input("EquivDiameter: ")) / original_df["EquivDiameter"].abs().max()
)
Extent = float(input("Extent: ")) / original_df["Extent"].abs().max()
Perimeter = float(input("Perimeter: ")) / original_df["Perimeter"].abs().max()
Roundness = float(input("Roundness: ")) / original_df["Roundness"].abs().max()
AspectRation = float(input("AspectRation: ")) / original_df["AspectRation"].abs().max()

my_inputs = [
    area,
    MajorAxisLength,
    MinorAxisLength,
    Eccentricity,
    ConvexArea,
    EquivDiameter,
    Extent,
    Perimeter,
    Roundness,
    AspectRation,
]

print("=" * 20)
model_inputs = torch.Tensor(my_inputs).to(device)
prediction = model(model_inputs)
print(prediction)
print("Class is: ", round(prediction.item()))
