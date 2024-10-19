import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Patch Embedding class for Vision Transformer (ViT).
    This class splits an input image into patches, flattens the patches, and projects them into a lower-dimensional space.

    Args:
        img_size (int): The size of the input image (default 224x224).
        patch_size (int): The size of each patch (default 16x16).
        in_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
        embed_dim (int): The dimensionality of the linear projection (embedding).
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        # Conv2d is used to project the patches into the embedding dimension.
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Forward pass for patch embedding.
        Splits the input image into patches, flattens them, and applies linear projection.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Projected patch embeddings of shape (batch_size, num_patches, embed_dim).
        """
        x = self.projection(x)  # Project patches into embedding space
        x = x.flatten(2)  # Flatten the patches
        x = x.transpose(1, 2)  # Transpose to get (batch_size, num_patches, embed_dim)
        return x


class PositionalEmbedding(nn.Module):
    """
    Positional Embedding class for ViT.
    This adds positional information to the patch embeddings to retain the spatial arrangement of the patches.

    Args:
        num_patches (int): The number of patches the image is divided into.
        embed_dim (int): The embedding dimension (same as the output of PatchEmbedding).
    """

    def __init__(self, num_patches, embed_dim):
        super(PositionalEmbedding, self).__init__()
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

    def forward(self, x):
        """
        Forward pass for adding positional embeddings.

        Args:
            x (Tensor): Patch embeddings of shape (batch_size, num_patches, embed_dim).

        Returns:
            Tensor: Patch embeddings with added positional information.
        """
        return x + self.pos_embedding


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder class for ViT.
    This encoder layer consists of multi-head self-attention followed by a feed-forward neural network (MLP).

    Args:
        embed_dim (int): Embedding dimension size (default 768).
        num_heads (int): Number of attention heads (default 12).
        hidden_dim (int): Hidden dimension of the feed-forward MLP (default 3072).
        dropout (float): Dropout rate for regularization (default 0.1).
    """

    def __init__(self, embed_dim=768, num_heads=12, hidden_dim=3072, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)  # Layer normalization for attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)  # Multi-head self-attention
        self.ln2 = nn.LayerNorm(embed_dim)  # Layer normalization for MLP
        # MLP consisting of two linear layers with GELU activation
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Forward pass through the Transformer encoder layer.

        Args:
            x (Tensor): Input embeddings of shape (batch_size, num_patches, embed_dim).

        Returns:
            Tensor: Processed embeddings of shape (batch_size, num_patches, embed_dim).
        """
        attn_output, _ = self.attention(x, x, x)  # Multi-head self-attention
        x = self.ln1(x + attn_output)  # Residual connection and layer norm
        x = self.ln2(x + self.mlp(x))  # Residual connection after MLP and layer norm
        return x


class ViT(nn.Module):
    """
    Vision Transformer (ViT) class for image classification.

    Args:
        img_size (int): The size of the input image (default 224).
        patch_size (int): The size of the patches the image is divided into (default 16).
        num_classes (int): The number of output classes for classification (default 3).
        embed_dim (int): Embedding dimension for patch embeddings (default 768).
        num_heads (int): Number of attention heads in the transformer (default 12).
        num_layers (int): Number of transformer encoder layers (default 12).
    """

    def __init__(self, img_size=224, patch_size=16, num_classes=3, embed_dim=768, num_heads=12, num_layers=12):
        super(ViT, self).__init__()
        num_patches = (img_size // patch_size) ** 2  # Calculate number of patches from image size and patch size

        # Patch embedding layer
        self.patch_embedding = PatchEmbedding(img_size, patch_size, embed_dim=embed_dim)

        # Positional embedding layer
        self.pos_embedding = PositionalEmbedding(num_patches, embed_dim)

        # Classification token (learned token for classification)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer encoders (stacked layers)
        self.transformer = nn.ModuleList([TransformerEncoder(embed_dim, num_heads) for _ in range(num_layers)])

        # Layer normalization
        self.ln = nn.LayerNorm(embed_dim)

        # MLP head for classification
        self.mlp_head = nn.Sequential(nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        """
        Forward pass for Vision Transformer.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tensor: Logits for classification of shape (batch_size, num_classes).
        """
        batch_size = x.shape[0]

        # Create patch embeddings
        x = self.patch_embedding(x)

        # Add classification token to the sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Expand cls token for batch size
        x = torch.cat((cls_tokens, x), dim=1)  # Concatenate cls token to patch embeddings

        # Add positional embedding
        x = self.pos_embedding(x)

        # Pass through transformer layers
        for layer in self.transformer:
            x = layer(x)

        # Use the CLS token output for classification
        cls_output = x[:, 0]
        return self.mlp_head(self.ln(cls_output))  # Apply final MLP head to classify

