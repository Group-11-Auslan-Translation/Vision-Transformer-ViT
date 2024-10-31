import torch
from vit import ViT  # Replace with the actual import path to your ViT model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = ViT(num_classes=10).to(device)

# Dummy input for the model (for example: grayscale image with size 224x224)
dummy_input = torch.randn(1, 1, 224, 224).to(device)

# Path to save the model in ONNX format
onnx_model_path = "vit_model.onnx"

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, onnx_model_path, export_params=True,
                  opset_version=12, do_constant_folding=True,
                  input_names=['input'], output_names=['output'])

print(f"Model saved to {onnx_model_path}")
