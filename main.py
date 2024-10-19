import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Splits an image into patches and creates linear embeddings from those patches.
    The input image is divided into fixed-size patches, flattened, and projected
    into a lower-dimensional embedding space.

    Args:
        img_size (int): The size of the input image
        patch_size (int): The size of each image patch
        in_channels (int): Number of input image channels
        embed_dim (int): Dimensionality of the linear patch embeddings.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        # convolutional layer to "flatten" patches and project them to embedding space
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Input x shape: [batch_size, in_channels, img_size, img_size]
        # Output x shape: [batch_size, embed_dim, num_patches, num_patches]
        x = self.projection(x)
        # Flatten the spatial dimensions and prepare for input into the transformer
        x = x.flatten(2)  # Shape: [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # Shape: [batch_size, num_patches, embed_dim]
        return x

class PositionalEmbedding(nn.Module):
    """
    Adds positional embeddings to the flattened image patches to retain spatial information.

    Args:
        num_patches (int): Number of patches in the image.
        embed_dim (int): Dimensionality of the linear patch embeddings.
    """
    def __init__(self, num_patches, embed_dim):
        super(PositionalEmbedding, self).__init__()
        # Positional embedding parameter (learned during training)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        # Add positional embeddings to the patch embeddings
        return x + self.pos_embedding

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder block, which processes the sequence of image patches.

    Consists of:
    - Multi-headed self-attention layer
    - Feed-forward neural network

    Args:
        embed_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimensionality of the hidden layer in the feed-forward network.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, embed_dim=768, num_heads=12, hidden_dim=3072, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        # LayerNorm and Multi-head attention
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # LayerNorm and MLP
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Apply multi-head attention
        attn_output, _ = self.attention(x, x, x)
        x = self.ln1(x + attn_output)  # Residual connection
        # Apply feed-forward network
        x = self.ln2(x + self.mlp(x))  # Residual connection
        return x

class ViT(nn.Module):
    """
    Vision Transformer (ViT) model for image classification.
    This model follows the overall structure of the Vision Transformer architecture:
    1. Split an image into patches (fixed sizes).
    2. Flatten the image patches.
    3. Create lower-dimensional linear embeddings from these flattened image patches.
    4. Include positional embeddings to retain spatial information.
    5. Feed the sequence as an input to a Transformer Encoder.
    6. Pre-train the ViT model on a large dataset (e.g., ImageNet), and fine-tune on downstream tasks.

    Args:
        img_size (int): The size of the input image (assumed to be square).
        patch_size (int): The size of each image patch (assumed to be square).
        num_classes (int): Number of classes for classification.
        embed_dim (int): Dimensionality of the patch embeddings.
        num_heads (int): Number of attention heads in the transformer encoder.
        num_layers (int): Number of transformer encoder layers.
    """
    def __init__(self, img_size=224, patch_size=16, num_classes=10, embed_dim=768, num_heads=12, num_layers=12):
        super(ViT, self).__init__()
        num_patches = (img_size // patch_size) ** 2  # Number of patches in the image
        # Patch embedding and positional embedding layers
        self.patch_embedding = PatchEmbedding(img_size, patch_size, embed_dim=embed_dim)
        self.pos_embedding = PositionalEmbedding(num_patches, embed_dim)
        # Classification token [CLS] to represent the entire image
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer encoder layers
        self.transformer = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # LayerNorm and MLP for classification
        self.ln = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, num_classes)  # Output: classification scores for num_classes
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # Patch embedding and adding [CLS] token
        x = self.patch_embedding(x)  # Shape: [batch_size, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: [batch_size, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # Concatenate CLS token to patch embeddings
        x = self.pos_embedding(x)  # Add positional embedding

        # Pass through transformer encoder layers
        for layer in self.transformer:
            x = layer(x)

        # Extract the [CLS] token output and pass it to the MLP head for classification
        cls_output = x[:, 0]
        return self.mlp_head(self.ln(cls_output))
