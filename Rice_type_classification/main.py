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
from sklearn.preprocessing import LabelEncoder
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


# 3. Label Encoding - WAŻNE: Konwertuj stringi na liczby
label_encoder = LabelEncoder()
data_df["Class"] = label_encoder.fit_transform(data_df["Class"])
num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")
print(f"Class mapping: {dict(zip(label_encoder.classes_, range(num_classes)))}")


# 4. Data Preprocessing
original_df = data_df.copy()
# Creating a copy of the original Dataframe to use to normalize inference

features_columns = data_df.columns[:-1]
for column in data_df.columns:
    data_df[column] = data_df[column] / data_df[column].abs().max()

print(data_df.head())


# 5. Data Splitting
X = np.array(
    data_df.iloc[:, :-1]
)  # Get the inputs, all rows and all columns except last column (output)
Y = np.array(
    data_df.iloc[:, -1]
)  # Get the ouputs, all rows and last column only (output column)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)  # Create the training split
X_test, X_val, Y_test, Y_val = train_test_split(
    X_test, Y_test, test_size=0.5, random_state=42
)  # Create the validation split
print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

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


# 6. Dataset Object
class dataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.Y = torch.tensor(Y, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


training_data = dataset(X_train, Y_train)
validation_data = dataset(X_val, Y_val)
testing_data = dataset(X_test, Y_test)

# 7. Training Hyper Parameters
BATCH_SIZE = 32
EPOCHS = 10
HIDDEN_UNITS = 32
LEARNING_RATE = 1e-3

# 8. Data Loaders
training_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)
testing_dataloader = DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=True)

# 9. Model Class


class RiceClassifcationModel(nn.Module):
    def __init__(self, input_size, hidden_units, num_classes):
        super(RiceClassifcationModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, num_classes),
        )

    def forward(self, x):
        return self.network(x)


# 10. Model Initialization
model = RiceClassifcationModel(X.shape[1], HIDDEN_UNITS, num_classes).to(device)
print(f"Model architecture:")
print(model)


# 11. Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)


# 12. Training Loop
total_loss_train_plot = []
total_loss_validation_plot = []
total_acc_train_plot = []
total_acc_validation_plot = []

for epoch in range(EPOCHS):
    # Training phase
    total_acc_train = 0
    total_loss_train = 0
    total_acc_val = 0
    total_loss_val = 0

    for data in training_dataloader:
        inputs, labels = data

        # Forward pass
        predictions = model(inputs)
        batch_loss = criterion(predictions, labels)

        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # Metrics
        total_loss_train += batch_loss.item()
        _, predicted = torch.max(predictions.data, 1)
        total_acc_train += (predicted == labels).sum().item()

    # Validation phase
    model.eval()
    total_acc_val = 0
    total_loss_val = 0

    with torch.no_grad():
        for data in validation_dataloader:
            inputs, labels = data
            predictions = model(inputs)
            batch_loss = criterion(predictions, labels)

            total_loss_val += batch_loss.item()
            _, predicted = torch.max(predictions.data, 1)
            total_acc_val += (predicted == labels).sum().item()

    # Store metrics
    avg_train_loss = total_loss_train / len(training_dataloader)
    avg_val_loss = total_loss_val / len(validation_dataloader)
    train_acc = (total_acc_train / len(training_data)) * 100
    val_acc = (total_acc_val / len(validation_data)) * 100

    total_loss_train_plot.append(avg_train_loss)
    total_acc_train_plot.append(train_acc)
    total_loss_validation_plot.append(avg_val_loss)
    total_acc_validation_plot.append(val_acc)

    print(
        f"Epoch {epoch + 1:2d}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%"
    )


# 13. Testing the Model

with torch.no_grad():
    total_acc_test = 0
    total_loss_test = 0

    for data in testing_dataloader:
        inputs, labels = data
        predictions = model(inputs)
        batch_loss_test = criterion(predictions, labels)

        total_loss_test += batch_loss_test.item()
        _, predicted = torch.max(predictions.data, 1)
        total_acc_test += (predicted == labels).sum().item()

print(f"Accuracy Score is: {round(total_acc_test / X_test.shape[0])*100, 2}%")

# 14. Plotting and Visualization

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


# 15. Inference
def predict_rice_type(
    area,
    major_axis,
    minor_axis,
    eccentricity,
    convex_area,
    equiv_diameter,
    extent,
    perimeter,
    roundness,
    aspect_ratio,
):
    # Normalize inputs using original data statistics
    normalized_inputs = [
        area / original_df["Area"].abs().max(),
        major_axis / original_df["MajorAxisLength"].abs().max(),
        minor_axis / original_df["MinorAxisLength"].abs().max(),
        eccentricity / original_df["Eccentricity"].abs().max(),
        convex_area / original_df["ConvexArea"].abs().max(),
        equiv_diameter / original_df["EquivDiameter"].abs().max(),
        extent / original_df["Extent"].abs().max(),
        perimeter / original_df["Perimeter"].abs().max(),
        roundness / original_df["Roundness"].abs().max(),
        aspect_ratio / original_df["AspectRation"].abs().max(),
    ]

    model.eval()
    with torch.no_grad():
        model_inputs = torch.tensor(normalized_inputs, dtype=torch.float32).to(device)
        predictions = model(model_inputs.unsqueeze(0))  # Add batch dimension
        _, predicted_class = torch.max(predictions, 1)

        # Convert back to original class name
        class_name = label_encoder.inverse_transform([predicted_class.item()])[0]
        confidence = torch.softmax(predictions, 1).max().item()

        return class_name, confidence


# Example usage (uncomment to use):
# result, confidence = predict_rice_type(15231, 421.91, 253.31, 0.83, 15617,
#                                       440.11, 0.73, 1360.73, 0.65, 1.67)
# print(f"Predicted class: {result} (confidence: {confidence:.2f})")
