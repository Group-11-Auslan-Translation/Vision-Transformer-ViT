import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from vit import ViT
from train import train_model, evaluate_model
from custom_dataset import CustomImageDataset
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Device configuration (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
num_epochs = 20
batch_size = 4
learning_rate = 1e-4
num_classes = 10

# Define the transformation for images
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),              # Resize images to 224x224 pixels
#     transforms.RandomHorizontalFlip(),          # Randomly flip images horizontally
#     transforms.RandomRotation(10),              # Randomly rotate images by up to 10 degrees
#     transforms.ToTensor(),                      # Convert images to PyTorch tensors
#     transforms.Normalize((0.5,), (0.5,))        # Normalize image pixel values (mean=0.5, std=0.5)
# ])

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 pixels
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop with scaling
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip horizontally
    transforms.RandomRotation(10),  # Randomly rotate by up to 10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jittering
    transforms.ToTensor(),  # Convert to tensor first
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),  # Random erasing (Cutout)
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Gaussian blur
    transforms.Normalize((0.5,), (0.5,)),  # Normalize image pixel values
])

# Dataset Path
dataset_path = '/content/drive/MyDrive/ViT/data/'

# dataset for training and validation
# dataset = CustomImageDataset(img_dir=dataset_path, transform=transform, limit_per_class = 600)
dataset = CustomImageDataset(img_dir=dataset_path, transform=transform)

# Check if dataset is empty
if len(dataset) == 0:
    raise ValueError(
        "The dataset is empty. Please check the dataset path and ensure there are images in the folders '0', '1', '2'.")

# Train-validation split
val_split = 0.2  # 20% of the data will be used for validation
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size

# Collect labels and print progress
all_labels = []
for i, (_, label) in enumerate(dataset):
    all_labels.append(label)

    # Print progress every 100 labels processed
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1} labels")

# After collecting all labels, print the total count
print(f"Total labels processed: {len(all_labels)}")

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)

# Print the class weights to check their values
print(f"Class weights: {class_weights}")

# Convert class weights to a tensor and move to the device (GPU/CPU)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Print the final tensor for confirmation
print(f"Class weights as tensor: {class_weights}")

# Ensure there are enough images to split into training and validation sets
if train_size == 0 or val_size == 0:
    raise ValueError(
        "Not enough data to split into training and validation sets. Please ensure the dataset has enough images.")

# Split the dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the ViT model
model = ViT(num_classes=num_classes).to(device)

# Loss function (Cross-Entropy Loss for classification)
# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(weight=class_weights)

# # Initialize the ViT model (Using pre-trained model from timm)
# from timm import create_model
# model = create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes).to(device)

# # Loss function (Cross-Entropy Loss with class weights)
# criterion = nn.CrossEntropyLoss(weight=class_weights)


# Optimizer (Adam optimizer for model parameters)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_losses, val_losses, train_accuracies, val_accuracies, train_precisions, val_precisions, train_recalls, val_recalls, train_f1_scores, val_f1_scores = train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, device
)

# Plot training metrics
plt.figure(figsize=(12, 8))

# Plot training losses
plt.subplot(2, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='b')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Progress')
plt.legend()

# Plot training accuracies
plt.subplot(2, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy', color='b')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Progress')
plt.legend()

# Plot training precisions
plt.subplot(2, 2, 3)
plt.plot(range(1, num_epochs + 1), train_precisions, label='Training Precision', color='b')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Training Precision Progress')
plt.legend()

# Plot training recalls
plt.subplot(2, 2, 4)
plt.plot(range(1, num_epochs + 1), train_recalls, label='Training Recall', color='b')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Training Recall Progress')
plt.legend()

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

# Plot validation metrics
plt.figure(figsize=(12, 8))

# Plot validation losses
plt.subplot(2, 2, 1)
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='g')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Progress')
plt.legend()

# Plot validation accuracies
plt.subplot(2, 2, 2)
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', color='g')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Progress')
plt.legend()

# Plot validation precisions
plt.subplot(2, 2, 3)
plt.plot(range(1, num_epochs + 1), val_precisions, label='Validation Precision', color='g')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Validation Precision Progress')
plt.legend()

# Plot validation recalls
plt.subplot(2, 2, 4)
plt.plot(range(1, num_epochs + 1), val_recalls, label='Validation Recall', color='g')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Validation Recall Progress')
plt.legend()

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

# Test dataset and evaluation
test_dataset = CustomImageDataset(img_dir=dataset_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate the model on the test set
evaluate_model(model, test_loader, device)
