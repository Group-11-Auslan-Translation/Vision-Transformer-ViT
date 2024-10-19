import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, limit_per_class=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_filenames = []
        self.labels = []

        # Loop through each folder named '0', '1', '2', etc., to gather image paths and labels
        for label in sorted(os.listdir(img_dir)):
            folder_path = os.path.join(img_dir, label)
            if os.path.isdir(folder_path) and label.isdigit():
                files = [file_name for file_name in os.listdir(folder_path) if
                         file_name.endswith('.jpg') or file_name.endswith('.png')]
                if limit_per_class:
                    files = files[:limit_per_class]
                for file_name in files:
                    self.image_filenames.append(os.path.join(folder_path, file_name))
                    self.labels.append(int(label))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = self.image_filenames[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset_path = 'dataset'
    dataset = CustomImageDataset(img_dir=dataset_path, transform=transform)
    print(f"Number of images in dataset: {len(dataset)}")
    image, label = dataset[0]
    print(f"Image shape: {image.shape}, Label: {label}")
