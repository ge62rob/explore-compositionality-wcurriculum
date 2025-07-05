import torch
from torch import nn

from config import NUM_PATCHES, EMBED_DIM

class PositionalEncoding(nn.Module):
    """
    Simple learnable embedding for patch positions.
    We have at most 256 patches, so we create an Embedding(256, embed_dim).
    """
    def __init__(self, embed_dim, max_positions=NUM_PATCHES):
        super().__init__()
        self.position_embedding = nn.Embedding(max_positions, embed_dim)

    def forward(self, x):
        """
        x: [batch_size, num_patches, embed_dim]
        Return: x + positional_embedding
        """
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        pos_emb = self.position_embedding(positions)  # [batch_size, seq_len, embed_dim]
        return x + pos_emb

class AutoregressiveTransformer(nn.Module):
    def __init__(self,
                 embed_dim=768,
                 num_heads=12,
                 num_layers=6,
                 num_patches=NUM_PATCHES,
                 dim_feedforward=3072,
                 dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        # Linear projection from patch_size^2 => embed_dim
        self.patch_embedding = nn.Linear(embed_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_positions=num_patches)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # default is seq_first => (S, N, E)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer: raw logits for each pixel => shape [patch_size^2]
        self.output_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask):
        """
        x: [batch_size, num_patches, embed_dim]  (flattened patch pixels)
        mask: [batch_size, num_patches] of booleans (True=masked patch).
        Returns predicted patches for the masked positions:
          shape [total_masked_patches, embed_dim] => raw logits.
        """
        # 1) Embed patches
        x = self.patch_embedding(x)  # [B, num_patches, embed_dim]

        # 2) Add positional encoding
        x = self.positional_encoding(x)  # [B, num_patches, embed_dim]

        # 3) Transformer wants shape [seq_len, batch, embed_dim]
        x = x.permute(1, 0, 2)  # => [num_patches, B, embed_dim]

        # 4) Create a causal mask so the model can only attend to previous patches
        causal_mask = torch.triu(torch.ones(self.num_patches, self.num_patches, device=x.device), diagonal=1).bool()

        # 5) Forward through Transformer => shape [num_patches, B, embed_dim]
        transformer_output = self.transformer_encoder(x, mask=causal_mask)

        # 6) Permute back => [B, num_patches, embed_dim]
        transformer_output = transformer_output.permute(1, 0, 2)

        # 7) Gather outputs for masked patches only => [sum(mask), embed_dim]
        masked_output = transformer_output[mask]

        # 8) Final linear -> patch_size^2 => raw logits
        predictions = self.output_layer(masked_output)  # => [sum(mask), embed_dim]
        return predictions 
