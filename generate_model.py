import torch
from torchviz import make_dot
from vit import ViT  # Import your ViT model

# Device configuration (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the ViT model # Adjust num_classes
model = ViT(num_classes=10).to(device)

# Create a dummy input tensor to simulate a batch of images (e.g., grayscale image with size 224x224)
dummy_input = torch.randn(1, 1, 224, 224).to(device)  # (batch_size, channels, height, width)

# Forward pass through the model to trace the computational graph
output = model(dummy_input)

# Use make_dot to visualize the model architecture and computation graph
model_viz = make_dot(output, params=dict(list(model.named_parameters()) + [('input', dummy_input)]))

# Save the diagram as a PNG file
model_viz.render("ViT_model_diagram", format="png")

print("Model diagram generated and saved as 'ViT_model_diagram.png'.")
