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
INPUT_SHAPE = 128 * 128 * 3
HIDDEN_UNITS = 128
OUTPUT_SHAPE = 3
LOSS_FN = nn.CrossEntropyLoss()
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
all_labels = data_df["category"]
label_encoder.fit(all_labels)
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        # transforms.ConvertImageDtype(torch.float),
    ]
)

# 9. Custom Dataset Class


class CustomDatasetClass(Dataset):
    def __init__(self, dataframe, transform=None, label_encoder=None):
        self.dataframe = dataframe
        self.transform = transform
        self.label_encoder = label_encoder

        # Przekształć etykiety dla tego datasetu
        if label_encoder:
            self.labels = torch.tensor(label_encoder.transform(dataframe["category"]))
        else:
            self.labels = torch.tensor(dataframe["category"].values)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe.iloc[index, 0]
        label = self.labels[index]

        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:  # Używaj self.transform zamiast globalnej zmiennej
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Zwróć domyślny obraz w przypadku błędu
            default_image = torch.zeros((3, 128, 128))
            return default_image, label


# 10. Datasets

train_dataset = CustomDatasetClass(train, transform)
test_dataset = CustomDatasetClass(test, transform)


# 11. Visualize data


def visualize_data(dataset, num_images=9):
    f, axarr = plt.subplots(N_ROWS, N_COLS, figsize=(10, 10))

    # Pobierz nazwy klas
    class_names = label_encoder.classes_

    for row in range(N_ROWS):
        for col in range(N_COLS):
            idx = np.random.randint(0, len(dataset))
            image, label = dataset[idx]

            # Przenieś obraz na CPU i przekształć do formatu do wyświetlenia
            image_np = image.cpu().permute(1, 2, 0).numpy()

            axarr[row, col].imshow(image_np)
            axarr[row, col].set_title(f"Class: {class_names[label]}")
            axarr[row, col].axis("off")

    plt.tight_layout()
    plt.show()


visualize_data(train_dataset)


# 12. Dataloaders


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# 13. Model


# 13.1. FC Model
class ImageClassificationModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=hidden_units, out_features=hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=hidden_units // 2, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# 13.2. CNN Model
import torch.nn.functional as F


class CNNImageClassifier(nn.Module):
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
        # Po 3 pooling: 128x128 -> 64x64 -> 32x32 -> 16x16
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# 14 Train and test loops (POPRAWIONE)
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
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Calculate accuracy
        if accuracy_fn:
            train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # 4. Optimizer zero grad
        optimizer.zero_grad()

        # 5. Backward pass
        loss.backward()

        # 6. Optimizer step
        optimizer.step()

    # Oblicz średnie
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    print(f"Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.2f}%")
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
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y).item()
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        # Oblicz średnie
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.2f}%")
    return test_loss, test_acc


# 14.2. Ulepszone funkcje trenowania z historią
def train_step_with_history(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn,
    device: torch.device = device,
):
    train_loss, train_acc = 0, 0
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Loss i accuracy
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Średnie
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    return train_loss, train_acc


def test_step_with_history(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    accuracy_fn,
    device: torch.device = device,
):
    test_loss, test_acc = 0, 0
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            test_pred = model(X)
            test_loss += loss_fn(test_pred, y).item()
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

    return test_loss, test_acc


# 15. Inicjalizacja obu modeli
print("\n" + "=" * 60)
print("PORÓWNANIE MODELI: FC vs CNN")
print("=" * 60)

# FC Model (twój oryginalny)
fc_model = ImageClassificationModel(
    input_shape=INPUT_SHAPE, hidden_units=HIDDEN_UNITS, output_shape=OUTPUT_SHAPE
).to(device)

# CNN Model
cnn_model = CNNImageClassifier(num_classes=OUTPUT_SHAPE).to(device)

# Optimizers
fc_optimizer = Adam(fc_model.parameters(), lr=LEARNING_RATE)
cnn_optimizer = Adam(cnn_model.parameters(), lr=LEARNING_RATE)

# Porównanie liczby parametrów
fc_params = sum(p.numel() for p in fc_model.parameters())
cnn_params = sum(p.numel() for p in cnn_model.parameters())

print(f"\nFC Model parameters: {fc_params:,}")
print(f"CNN Model parameters: {cnn_params:,}")
print(f"CNN ma {cnn_params/fc_params:.2f}x więcej parametrów")


# 16. Training obu modeli
fc_history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
cnn_history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

print(f"\n{'='*30} FC MODEL TRAINING {'='*30}")
for epoch in range(EPOCHS):
    print(f"\nFC Epoch: {epoch+1}/{EPOCHS}")
    print("-" * 40)

    train_loss, train_acc = train_step_with_history(
        fc_model, train_loader, LOSS_FN, fc_optimizer, accuracy_fn, device
    )
    test_loss, test_acc = test_step_with_history(
        fc_model, test_loader, LOSS_FN, accuracy_fn, device
    )

    # Zapisz historię
    fc_history["train_loss"].append(train_loss)
    fc_history["train_acc"].append(train_acc)
    fc_history["test_loss"].append(test_loss)
    fc_history["test_acc"].append(test_acc)

    print(f"FC Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.2f}%")
    print(f"FC Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.2f}%")


print(f"\n{'='*30} CNN MODEL TRAINING {'='*30}")
for epoch in range(EPOCHS):
    print(f"\nCNN Epoch: {epoch+1}/{EPOCHS}")
    print("-" * 40)

    train_loss, train_acc = train_step_with_history(
        cnn_model, train_loader, LOSS_FN, cnn_optimizer, accuracy_fn, device
    )
    test_loss, test_acc = test_step_with_history(
        cnn_model, test_loader, LOSS_FN, accuracy_fn, device
    )

    # Zapisz historię
    cnn_history["train_loss"].append(train_loss)
    cnn_history["train_acc"].append(train_acc)
    cnn_history["test_loss"].append(test_loss)
    cnn_history["test_acc"].append(test_acc)

    print(f"CNN Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.2f}%")
    print(f"CNN Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.2f}%")


# 17. Porównanie wyników
print("\n" + "=" * 60)
print("PODSUMOWANIE WYNIKÓW")
print("=" * 60)

fc_best_acc = max(fc_history["test_acc"])
cnn_best_acc = max(cnn_history["test_acc"])

fc_final_acc = fc_history["test_acc"][-1]
cnn_final_acc = cnn_history["test_acc"][-1]

print(f"\nFC Model:")
print(f"  Best Test Accuracy: {fc_best_acc:.2f}%")
print(f"  Final Test Accuracy: {fc_final_acc:.2f}%")
print(f"  Final Train Loss: {fc_history['train_loss'][-1]:.4f}")

print(f"\nCNN Model:")
print(f"  Best Test Accuracy: {cnn_best_acc:.2f}%")
print(f"  Final Test Accuracy: {cnn_final_acc:.2f}%")
print(f"  Final Train Loss: {cnn_history['train_loss'][-1]:.4f}")

print(f"\nImprovement: {cnn_final_acc - fc_final_acc:+.2f}% accuracy gain with CNN")


# 18. Wizualizacja porównania
def plot_training_comparison():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    epochs_range = range(1, EPOCHS + 1)

    # Training Loss
    ax1.plot(
        epochs_range, fc_history["train_loss"], "b-", label="FC Model", linewidth=2
    )
    ax1.plot(
        epochs_range, cnn_history["train_loss"], "r-", label="CNN Model", linewidth=2
    )
    ax1.set_title("Training Loss Comparison")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Test Loss
    ax2.plot(epochs_range, fc_history["test_loss"], "b-", label="FC Model", linewidth=2)
    ax2.plot(
        epochs_range, cnn_history["test_loss"], "r-", label="CNN Model", linewidth=2
    )
    ax2.set_title("Test Loss Comparison")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Training Accuracy
    ax3.plot(epochs_range, fc_history["train_acc"], "b-", label="FC Model", linewidth=2)
    ax3.plot(
        epochs_range, cnn_history["train_acc"], "r-", label="CNN Model", linewidth=2
    )
    ax3.set_title("Training Accuracy Comparison")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy (%)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Test Accuracy
    ax4.plot(epochs_range, fc_history["test_acc"], "b-", label="FC Model", linewidth=2)
    ax4.plot(
        epochs_range, cnn_history["test_acc"], "r-", label="CNN Model", linewidth=2
    )
    ax4.set_title("Test Accuracy Comparison")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy (%)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Wywołaj wizualizację
plot_training_comparison()


# 19. Test pojedynczych predykcji
def test_random_predictions(fc_model, cnn_model, test_dataset, num_samples=6):
    """Testuje oba modele na losowych próbkach"""

    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))

    # Nazwy klas (dostosuj do swoich danych)
    class_names = ["Healthy", "Disease_1", "Disease_2"]  # Zmień na prawdziwe nazwy

    fc_model.eval()
    cnn_model.eval()

    with torch.no_grad():
        for i in range(num_samples):
            # Losowa próbka
            idx = np.random.randint(0, len(test_dataset))
            image, true_label = test_dataset[idx]

            # Dodaj batch dimension
            image_batch = image.unsqueeze(0).to(device)

            # Predykcje
            fc_pred = fc_model(image_batch).argmax(dim=1).cpu().item()
            cnn_pred = cnn_model(image_batch).argmax(dim=1).cpu().item()

            # Wyświetl obraz
            img_display = image.cpu().permute(1, 2, 0).numpy()
            if img_display.min() < 0:  # Jeśli znormalizowane
                img_display = (img_display - img_display.min()) / (
                    img_display.max() - img_display.min()
                )

            # FC predictions
            axes[0, i].imshow(img_display)
            axes[0, i].set_title(
                f"FC: {class_names[fc_pred]}\nTrue: {class_names[true_label]}"
            )
            axes[0, i].axis("off")
            if fc_pred == true_label:
                axes[0, i].add_patch(
                    plt.Rectangle((0, 0), 127, 127, fill=False, edgecolor="green", lw=3)
                )
            else:
                axes[0, i].add_patch(
                    plt.Rectangle((0, 0), 127, 127, fill=False, edgecolor="red", lw=3)
                )

            # CNN predictions
            axes[1, i].imshow(img_display)
            axes[1, i].set_title(
                f"CNN: {class_names[cnn_pred]}\nTrue: {class_names[true_label]}"
            )
            axes[1, i].axis("off")
            if cnn_pred == true_label:
                axes[1, i].add_patch(
                    plt.Rectangle((0, 0), 127, 127, fill=False, edgecolor="green", lw=3)
                )
            else:
                axes[1, i].add_patch(
                    plt.Rectangle((0, 0), 127, 127, fill=False, edgecolor="red", lw=3)
                )

    axes[0, 0].set_ylabel("FC Model", fontsize=14, fontweight="bold")
    axes[1, 0].set_ylabel("CNN Model", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()


# 20. Confusion Matrix dla obu modeli
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def evaluate_both_models(fc_model, cnn_model, test_loader):
    """Szczegółowa ewaluacja obu modeli"""

    # Zbierz predykcje dla obu modeli
    fc_preds, cnn_preds, true_labels = [], [], []

    fc_model.eval()
    cnn_model.eval()

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            # FC predictions
            fc_outputs = fc_model(X)
            fc_pred = fc_outputs.argmax(dim=1)

            # CNN predictions
            cnn_outputs = cnn_model(X)
            cnn_pred = cnn_outputs.argmax(dim=1)

            fc_preds.extend(fc_pred.cpu().numpy())
            cnn_preds.extend(cnn_pred.cpu().numpy())
            true_labels.extend(y.cpu().numpy())

    # Nazwy klas
    class_names = ["Healthy", "Disease_1", "Disease_2"]  # Dostosuj

    # Confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # FC Confusion Matrix
    cm_fc = confusion_matrix(true_labels, fc_preds)
    sns.heatmap(
        cm_fc,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax1,
    )
    ax1.set_title("FC Model - Confusion Matrix")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")

    # CNN Confusion Matrix
    cm_cnn = confusion_matrix(true_labels, cnn_preds)
    sns.heatmap(
        cm_cnn,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax2,
    )
    ax2.set_title("CNN Model - Confusion Matrix")
    ax2.set_ylabel("True Label")
    ax2.set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.show()

    # Classification reports
    print("\n" + "=" * 40)
    print("FC MODEL - CLASSIFICATION REPORT")
    print("=" * 40)
    print(classification_report(true_labels, fc_preds, target_names=class_names))

    print("\n" + "=" * 40)
    print("CNN MODEL - CLASSIFICATION REPORT")
    print("=" * 40)
    print(classification_report(true_labels, cnn_preds, target_names=class_names))

    # Accuracy porównanie
    from sklearn.metrics import accuracy_score

    fc_acc = accuracy_score(true_labels, fc_preds) * 100
    cnn_acc = accuracy_score(true_labels, cnn_preds) * 100

    print(f"\n{'='*50}")
    print("FINAL COMPARISON")
    print(f"{'='*50}")
    print(f"FC Model Final Accuracy:  {fc_acc:.2f}%")
    print(f"CNN Model Final Accuracy: {cnn_acc:.2f}%")
    print(f"Improvement: {cnn_acc - fc_acc:+.2f}%")

    return fc_acc, cnn_acc


# 21. Uruchom CNN training
print(f"\n{'='*30} CNN MODEL TRAINING {'='*30}")

for epoch in range(EPOCHS):
    print(f"\nCNN Epoch: {epoch+1}/{EPOCHS}")
    print("-" * 40)

    train_loss, train_acc = train_step_with_history(
        cnn_model, train_loader, LOSS_FN, cnn_optimizer, accuracy_fn, device
    )
    test_loss, test_acc = test_step_with_history(
        cnn_model, test_loader, LOSS_FN, accuracy_fn, device
    )

    # Zapisz historię
    cnn_history["train_loss"].append(train_loss)
    cnn_history["train_acc"].append(train_acc)
    cnn_history["test_loss"].append(test_loss)
    cnn_history["test_acc"].append(test_acc)

    print(f"CNN Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.2f}%")
    print(f"CNN Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.2f}%")


# 22. Wizualizacje i porównania
print("\n" + "=" * 60)
print("GENEROWANIE WIZUALIZACJI...")
print("=" * 60)

# Wykres porównania
plot_training_comparison()

# Test na losowych próbkach
print("\nTesting random predictions...")
test_random_predictions(fc_model, cnn_model, test_dataset)

# Szczegółowa ewaluacja
print("\nDetailed evaluation...")
fc_final_acc, cnn_final_acc = evaluate_both_models(fc_model, cnn_model, test_loader)

# 23. Zapisz najlepszy model
best_model = cnn_model if cnn_final_acc > fc_final_acc else fc_model
model_type = "CNN" if cnn_final_acc > fc_final_acc else "FC"

torch.save(best_model.state_dict(), f"best_{model_type.lower()}_model.pth")
print(
    f"\n✅ Saved best model: {model_type} with {max(fc_final_acc, cnn_final_acc):.2f}% accuracy"
)
