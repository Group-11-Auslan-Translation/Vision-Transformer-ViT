import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from vit import ViT
from train import train_model, evaluate_model
from custom_dataset import CustomImageDataset
import matplotlib.pyplot as plt

# Device configuration (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
num_epochs = 20
batch_size = 8
learning_rate = 0.001
num_classes = 3

# Define the transformation for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),              # Resize images to 224x224 pixels
    transforms.RandomHorizontalFlip(),          # Randomly flip images horizontally
    transforms.RandomRotation(10),              # Randomly rotate images by up to 10 degrees
    transforms.ToTensor(),                      # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))        # Normalize image pixel values (mean=0.5, std=0.5)
])

# Dataset Path
dataset_path = '/dataset/'

# Full dataset for training and validation
dataset = CustomImageDataset(img_dir=dataset_path, transform=transform)

# Check if dataset is empty
if len(dataset) == 0:
    raise ValueError("The dataset is empty. Please check the dataset path and ensure there are images in the folders '0', '1', '2'.")

# Train-validation split
val_split = 0.2  # 20% of the data will be used for validation
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size

# Ensure there are enough images to split into training and validation sets
if train_size == 0 or val_size == 0:
    raise ValueError("Not enough data to split into training and validation sets. Please ensure the dataset has enough images.")

# Split the dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the ViT model
model = ViT(num_classes=num_classes).to(device)

# Loss function (Cross-Entropy Loss for classification)
criterion = nn.CrossEntropyLoss()

# Optimizer (Adam optimizer for model parameters)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_losses, val_accuracies, val_precisions, val_recalls, val_f1_scores = train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, device
)

# Plot training metrics
plt.figure(figsize=(16, 12))

# Plot training losses
plt.subplot(2, 3, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='b')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Progress')
plt.legend()

# Plot validation accuracies
plt.subplot(2, 3, 2)
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', color='g')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Progress')
plt.legend()

# Plot validation precisions
plt.subplot(2, 3, 3)
plt.plot(range(1, num_epochs + 1), val_precisions, label='Validation Precision', color='r')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Validation Precision Progress')
plt.legend()

# Plot validation recalls
plt.subplot(2, 3, 4)
plt.plot(range(1, num_epochs + 1), val_recalls, label='Validation Recall', color='m')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Validation Recall Progress')
plt.legend()

# Plot validation F1-scores
plt.subplot(2, 3, 5)
plt.plot(range(1, num_epochs + 1), val_f1_scores, label='Validation F1-Score', color='c')
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.title('Validation F1-Score Progress')
plt.legend()

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

# Test dataset and evaluation
test_dataset = CustomImageDataset(img_dir=dataset_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate the model on the test set
evaluate_model(model, test_loader, device)
