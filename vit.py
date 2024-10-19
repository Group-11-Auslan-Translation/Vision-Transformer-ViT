import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

    def forward(self, x):
        return x + self.pos_embedding

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, hidden_dim=3072, dropout=0.1):
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
        x = self.ln1(x + attn_output)
        x = self.ln2(x + self.mlp(x))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=3, embed_dim=768, num_heads=12, num_layers=12):
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
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embedding(x)
        for layer in self.transformer:
            x = layer(x)
        cls_output = x[:, 0]
        return self.mlp_head(self.ln(cls_output))
