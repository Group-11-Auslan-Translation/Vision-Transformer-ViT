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
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=256):  # Embed_dim reduced to 256
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        # convolutional layer to "flatten" patches and project them to embedding space
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)  # Shape: [batch_size, num_patches, embed_dim]
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))  # Adjust num_patches for CLS token

    def forward(self, x):
        return x + self.pos_embedding

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, hidden_dim=512, dropout=0.1):  # Simpler Transformer layer
        super(TransformerEncoder, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.ln1(x + attn_output)  # Residual connection
        x = self.ln2(x + self.mlp(x))  # Residual connection
        return x

class ViT(nn.Module):
    """
    Simplified Vision Transformer (ViT) model for image classification.
    Args:
        img_size (int): The size of the input image (assumed to be square).
        patch_size (int): The size of each image patch (assumed to be square).
        num_classes (int): Number of classes for classification.
        embed_dim (int): Dimensionality of the patch embeddings.
        num_heads (int): Number of attention heads in the transformer encoder.
        num_layers (int): Number of transformer encoder layers.
    """
    def __init__(self, img_size=224, patch_size=16, num_classes=10, embed_dim=256, num_heads=4, num_layers=4):  # Simplified model
        super(ViT, self).__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(img_size, patch_size, embed_dim=embed_dim)
        self.pos_embedding = PositionalEmbedding(num_patches, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))


        self.transformer = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads) for _ in range(num_layers)
        ])

        self.ln = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # Concatenate CLS token to patch embeddings
        x = self.pos_embedding(x)

        for layer in self.transformer:
            x = layer(x)

        cls_output = x[:, 0]
        return self.mlp_head(self.ln(cls_output))
