import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from main import ViT
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Check if GPU is available and use it, otherwise fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
num_epochs = 20  # Number of training epochs
batch_size = 4  # Batch size for training
learning_rate = 0.001  # Learning rate for the optimizer
num_classes = 10  # Number of classes (digits 0-9)
validation_split = 0.2

# Define transformations for grayscale and optional Gaussian noise (comment out noise if not needed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalization for single channel
])

# Load the dataset
dataset = datasets.ImageFolder(root='dataset/train', transform=transform)

# Split the dataset into training and validation sets
total_size = len(dataset)
val_size = int(total_size * validation_split)
train_size = total_size - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the Vision Transformer model, loss function, and optimizer
model = ViT(num_classes=num_classes).to(device)  # Move model to the chosen device (GPU/CPU)
criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

best_val_f1 = 0.0

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0  # Track the total loss for this epoch
    all_preds = []  # List to store predictions
    all_labels = []  # List to store ground truth labels

    # Loop through the training data
    for images, labels in train_loader:
        images = images.to(device)  # Move images to the device
        labels = labels.to(device)  # Move labels to the device

        optimizer.zero_grad()  # Clear the gradients of the optimizer
        outputs = model(images)  # Forward pass: compute model predictions
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backpropagate the gradients
        optimizer.step()  # Update the model parameters

        running_loss += loss.item()  # Accumulate the running loss

        # Convert outputs to predicted class labels
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())  # Store the predictions
        all_labels.extend(labels.cpu().numpy())  # Store the true labels

    # Calculate training metrics
    train_accuracy = accuracy_score(all_labels, all_preds)
    train_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    train_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_running_loss = 0.0  # Track validation loss
    val_preds = []  # List to store validation predictions
    val_labels = []  # List to store validation labels

    with torch.no_grad():  # Disable gradient calculation during validation
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    # Calculate validation metrics
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
    val_recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

    # Print training and validation metrics
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {running_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_accuracy:.4f} | "
          f"Train Precision: {train_precision:.4f} | "
          f"Train Recall: {train_recall:.4f} | "
          f"Train F1: {train_f1:.4f}")
    print(f"Val Loss: {val_running_loss/len(val_loader):.4f} | "
          f"Val Acc: {val_accuracy:.4f} | "
          f"Val Precision: {val_precision:.4f} | "
          f"Val Recall: {val_recall:.4f} | "
          f"Val F1: {val_f1:.4f}\n")

    # Save the model with the best validation F1 score
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'best_vit_auslan_model.pth')
        print("Best model saved.")

# Save the final trained model
torch.save(model.state_dict(), 'vit_custom_auslan_model.pth')
print("Training complete. Model saved as vit_custom_auslan_model.pth")
