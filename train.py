import os
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score
from main_vit import ViT

# Device configuration (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
num_epochs = 30
batch_size = 2  # Adjust
learning_rate = 1e-4
num_classes = 10

# Define the transformation for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalization for single channel images
])

# Custom Dataset for loading augmented images
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        self.image_filenames = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert('L')  # Open image as grayscale

        label = int(self.image_filenames[idx][0])
        if self.transform:
            image = self.transform(image)
        return image, label

# Load dataset from the augmented images folder
dataset = CustomImageDataset(img_dir='augmented_data/', transform=transform)

# Train-validation split
val_split = 0.2
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = ViT(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_train_preds = []
    all_train_labels = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(all_train_labels, all_train_preds)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.4f}")

    # Validation phase
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_vit_auslan_model.pth')
        print("Best model saved.")

# After training, load the best model and test it on the same augmented images
model.load_state_dict(torch.load('best_vit_auslan_model.pth', map_location=device))
model.eval()

# Test the model on the augmented images
all_preds = []
true_labels = [int(f[0]) for f in dataset.image_filenames]  # Infer true labels from filenames

for img_name in dataset.image_filenames:
    img_path = os.path.join('augmented_data/', img_name)
    image = Image.open(img_path).convert('L')  # Open image as grayscale
    image = transform(image).unsqueeze(0).to(device)  # Preprocess and add batch dimension

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        all_preds.append(pred.item())

# Display test predictions
for i, pred in enumerate(all_preds):
    print(f"Image {dataset.image_filenames[i]} - True Label: {true_labels[i]}, Predicted Label: {pred}")

# Calculate accuracy on test images
accuracy = sum([1 for i, pred in enumerate(all_preds) if pred == true_labels[i]]) / len(all_preds)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
