import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomImageDataset(Dataset):
    """
    Custom dataset for loading images from folders named with digits ('0', '1', '2', etc.).
    Each folder represents a class, and the images inside are labeled accordingly.

    Args:
        img_dir (str): Path to the directory containing the image folders.
        transform (callable, optional): A function/transform to apply to the images.
        limit_per_class (int, optional): Maximum number of images to load per class.
    """

    def __init__(self, img_dir, transform=None, limit_per_class=None):
        """
        Initialize the dataset by gathering image file paths and their corresponding labels.

        Args:
            img_dir (str): The directory where the image folders are stored.
            transform (callable, optional): Transformation to be applied to the images.
            limit_per_class (int, optional): A limit on the number of images to load per class.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.image_filenames = []
        self.labels = []

        # Loop through each folder (class) and collect image paths and labels
        for label in sorted(os.listdir(img_dir)):
            folder_path = os.path.join(img_dir, label)
            if os.path.isdir(folder_path) and label.isdigit():  # Ensure the folder name is a digit
                # Collect image files with '.jpg' or '.png' extension
                files = [file_name for file_name in os.listdir(folder_path)
                         if file_name.endswith('.jpg') or file_name.endswith('.png')]

                # Limit the number of files per class if needed
                if limit_per_class:
                    files = files[:limit_per_class]

                # Append each image path and its label (folder name) to the respective lists
                for file_name in files:
                    self.image_filenames.append(os.path.join(folder_path, file_name))
                    self.labels.append(int(label))

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Retrieve an image and its label by index.

        Args:
            idx (int): Index of the image.

        Returns:
            image (PIL Image or transformed image): The image at the specified index.
            label (int): The label corresponding to the image.
        """
        img_path = self.image_filenames[idx]
        image = Image.open(img_path).convert('L')  # Convert image to grayscale
        label = self.labels[idx]

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    # Define the transformations to be applied to each image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to have mean 0.5 and std 0.5
    ])

    dataset_path = '/content/drive/MyDrive/ViT/datas/'  # Path to the dataset
    dataset = CustomImageDataset(img_dir=dataset_path, transform=transform)

    # Output the total number of images and details of the first image
    print(f"Number of images in dataset: {len(dataset)}")
    image, label = dataset[0]
    print(f"Image shape: {image.shape}, Label: {label}")
