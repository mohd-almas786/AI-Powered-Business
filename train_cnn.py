import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from cnn_model import CustomerImageCNN


# 1. Configuration

RANDOM_STATE = 42
IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.pth")


# 2. Create Synthetic Image Dataset

def create_synthetic_images(num_samples=1000):
    """
    Creates simple synthetic grayscale images.

    Class 0 = Low-risk customer pattern
    Class 1 = High-risk customer pattern

    This is only for demonstrating the CNN pipeline.
    """

    np.random.seed(RANDOM_STATE)

    images = []
    labels = []

    for _ in range(num_samples):

        label = np.random.randint(0, 2)

        image = np.random.normal(
            loc=0.2 if label == 0 else 0.7,
            scale=0.15,
            size=(IMAGE_SIZE, IMAGE_SIZE)
        )

        image = np.clip(image, 0, 1)

        # Add a simple pattern based on the class
        if label == 1:
            image[20:45, 20:45] += 0.2
        else:
            image[5:20, 5:20] += 0.1

        image = np.clip(image, 0, 1)

        images.append(image)
        labels.append(label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # CNN expects:
    # [samples, channels, height, width]

    images = images[:, np.newaxis, :, :]

    return images, labels


# 3. Load Dataset

print("\nCreating synthetic image dataset...")

X, y = create_synthetic_images()

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)


# 4. Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 5. Convert to PyTorch Tensors

X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)

y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)


# 6. Create DataLoaders

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# 7. Create CNN Model

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nUsing device:", device)

model = CustomerImageCNN(num_classes=2)
model = model.to(device)


# 8. Loss Function and Optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# 9. Training

print("\nStarting CNN training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)

    epoch_accuracy = correct / total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.4f}"
    )


# 10. Evaluation

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


test_accuracy = correct / total

print("\n--------------------------------")
print("CNN Evaluation")
print("--------------------------------")

print(f"Test Accuracy: {test_accuracy:.4f}")


# 11. Save Model

os.makedirs(MODEL_DIR, exist_ok=True)

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\nCNN model saved successfully:")
print(MODEL_PATH)

print("\nPart 15 CNN training completed successfully!")