"""
Masked Autoencoder (MAE) model architecture
Based on Vision Transformer with masking and reconstruction
"""

import torch
import torch.nn as nn
import numpy as np



# ======PATCH EMBEDDING=======

class PatchEmbed(nn.Module):
    """
    Split image into patches and embed them
    
    Input: (B, C, H, W) - Batch of images
    Output: (B, num_patches, embed_dim) - Batch of patch embeddings
    """
    
    def __init__(self, img_size=256, patch_size=16, in_channels=15, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolutional layer to extract patches and embed them
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Apply convolution: (B, C, H, W) -> (B, embed_dim, num_patches_h, num_patches_w)
        x = self.proj(x)
        
        # Flatten: (B, embed_dim, num_patches_h, num_patches_w) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Transpose: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        return x



# ======TRANSFORMER BLOCK=======

class TransformerBlock(nn.Module):
    """
    Standard Transformer block with multi-head attention and MLP
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP (Feed-forward network)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x



# ======VISION TRANSFORMER ENCODER=======

class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder
    Processes visible (unmasked) patches
    """
    
    def __init__(
        self, 
        img_size=256, 
        patch_size=16, 
        in_channels=15,
        embed_dim=768, 
        depth=12, 
        num_heads=12,
        mlp_ratio=4.0
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Learnable positional embeddings (one for each patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize positional embeddings
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x, mask=None):
        # x: (B, C, H, W)
        
        # Patch embedding: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply mask if provided (keep only visible patches)
        if mask is not None:
            x = x[~mask].reshape(x.shape[0], -1, x.shape[-1])
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        return x



# DECODER

class MAEDecoder(nn.Module):
    """
    Lightweight decoder to reconstruct masked patches
    """
    
    def __init__(
        self,
        num_patches=256,
        patch_size=16,
        in_channels=15,
        encoder_embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0
    ):
        super().__init__()
        
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        # Project encoder output to decoder dimension
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        
        # Learnable mask token (used for masked patches)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Positional embeddings for decoder
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        
        # Transformer blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio)
            for _ in range(decoder_depth)
        ])
        
        # Final layer norm
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Prediction head: project back to pixel space
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, 
            patch_size ** 2 * in_channels
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
    
    def forward(self, x, ids_restore):
        # x: encoded visible patches (B, num_visible, encoder_embed_dim)
        # ids_restore: indices to unshuffle patches back to original order
        
        # Project to decoder dimension
        x = self.decoder_embed(x)
        
        # Append mask tokens to sequence
        B = x.shape[0]
        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] - x.shape[1], 1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        
        # Unshuffle to restore original order
        x_full = torch.gather(
            x_full, 
            dim=1, 
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        
        # Add positional embeddings
        x = x_full + self.decoder_pos_embed
        
        # Apply transformer blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        # Final normalization
        x = self.decoder_norm(x)
        
        # Predict pixel values
        x = self.decoder_pred(x)
        
        return x



# ======MASKED AUTOENCODER (MAE)=======

class MaskedAutoencoder(nn.Module):
    """
    Complete MAE model: Encoder + Decoder + Masking
    """
    
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_channels=15,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        mask_ratio=0.75
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.mask_ratio = mask_ratio
        
        # Encoder
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio
        )
        
        # Decoder
        num_patches = (img_size // patch_size) ** 2
        self.decoder = MAEDecoder(
            num_patches=num_patches,
            patch_size=patch_size,
            in_channels=in_channels,
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio
        )
    
    def random_masking(self, x):
        """
        Perform random masking by shuffling patches
        
        Args:
            x: (B, num_patches, embed_dim)
        
        Returns:
            x_masked: visible patches only
            mask: binary mask (True = masked, False = visible)
            ids_restore: indices to restore original order
        """
        B, N, D = x.shape  # batch, num_patches, embed_dim
        
        # Number of patches to keep (visible)
        num_keep = int(N * (1 - self.mask_ratio))
        
        # Random shuffle
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep only the first num_keep patches (visible)
        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def patchify(self, imgs):
        """
        Convert images to patches
        
        Args:
            imgs: (B, C, H, W)
        
        Returns:
            patches: (B, num_patches, patch_size**2 * C)
        """
        p = self.patch_size
        B, C, H, W = imgs.shape
        h = H // p
        w = W // p
        
        # Reshape to patches
        x = imgs.reshape(B, C, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(B, h * w, p ** 2 * C)
        
        return x
    
    def unpatchify(self, x):
        """
        Convert patches back to images
        
        Args:
            x: (B, num_patches, patch_size**2 * C)
        
        Returns:
            imgs: (B, C, H, W)
        """
        p = self.patch_size
        C = self.in_channels
        h = w = int(x.shape[1] ** 0.5)
        
        x = x.reshape(x.shape[0], h, w, p, p, C)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], C, h * p, w * p)
        
        return imgs
    
    def forward_loss(self, imgs, pred, mask):
        """
        Calculate reconstruction loss (MSE) on masked patches only
        
        Args:
            imgs: original images (B, C, H, W)
            pred: predicted patches (B, num_patches, patch_size**2 * C)
            mask: binary mask (B, num_patches) - 1 is masked, 0 is visible
        
        Returns:
            loss: mean loss on masked patches
        """
        target = self.patchify(imgs)
        
        # Calculate loss only on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean per patch
        loss = (loss * mask).sum() / mask.sum()  # Mean over masked patches
        
        return loss
    
    def forward(self, imgs):
        """
        Forward pass: encode -> mask -> decode -> reconstruct
        
        Args:
            imgs: (B, C, H, W)
        
        Returns:
            loss: reconstruction loss
            pred: predicted images
            mask: binary mask
        """
        # Encode patches
        latent = self.encoder.patch_embed(imgs)
        latent = latent + self.encoder.pos_embed
        
        # Random masking
        latent_masked, mask, ids_restore = self.random_masking(latent)
        
        # Encode visible patches
        for block in self.encoder.blocks:
            latent_masked = block(latent_masked)
        latent_masked = self.encoder.norm(latent_masked)
        
        # Decode and reconstruct
        pred = self.decoder(latent_masked, ids_restore)
        
        # Calculate loss
        loss = self.forward_loss(imgs, pred, mask)
        
        # Reconstruct full image for visualization
        pred_imgs = self.unpatchify(pred)
        
        return loss, pred_imgs, mask



# ======MODEL INITIALIZATION=======

def build_mae(config):
    """
    Build MAE model from config
    
    Args:
        config: configuration module with model parameters
    
    Returns:
        model: MAE model
    """
    model = MaskedAutoencoder(
        img_size=config.IMAGE_SIZE,
        patch_size=config.PATCH_SIZE,
        in_channels=config.IN_CHANNELS,
        encoder_embed_dim=config.EMBED_DIM,
        encoder_depth=config.ENCODER_DEPTH,
        encoder_num_heads=config.ENCODER_NUM_HEADS,
        decoder_embed_dim=config.DECODER_EMBED_DIM,
        decoder_depth=config.DECODER_DEPTH,
        decoder_num_heads=config.DECODER_NUM_HEADS,
        mlp_ratio=config.ENCODER_MLP_RATIO,
        mask_ratio=config.MASK_RATIO
    )
    
    return model


if __name__ == "__main__":
    # Quick test
    print("Testing MAE model...")
    
    model = MaskedAutoencoder(
        img_size=256,
        patch_size=16,
        in_channels=15,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mask_ratio=0.75
    )
    
    # Test forward pass
    x = torch.randn(2, 15, 256, 256)  # Batch of 2 images
    loss, pred, mask = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Masked patches: {mask.sum().item() / mask.numel():.1%}")
    print("\nModel is working correctly.")