import torch
from torch.nn import functional as F
import torch.nn as nn
from attention import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, H/8, W/8) -> (batch_size, 512, H/4, W/4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, H/4, W/4) -> (batch_size, 512, H/2, W/2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # (batch_size, 512, H/2, W/2) -> (batch_size, 256, H/2, W/2)
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # (batch_size, 256, H/2, W/2) -> (batch_size, 256, H, W)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            # (batch_size, 256, H, W) -> (batch_size, 128, H, W)
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128), # GroupNorm divide 128 into groups of 32

            nn.SiLU(),

            # (batch_size, 128, H, W) -> (batch_size, 3, W, H)
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (batch_size, 4, H/8, W/8)
        x /= 0.18215

        for module in self:
            x = module(x)

        # batch_size, 3, H, W
        return x
    

class VAE_Decoder_Optimized(nn.Module):
    def __init__(self):
        super().__init__()
        
        from attention import OptimizedBlock, OptimizedAttention, AdaptiveLayerNorm
        
        # Initial projection
        self.init_proj = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=1),
            nn.Conv2d(4, 512, kernel_size=3, padding=1)
        )
        
        # Middle blocks
        self.middle_blocks = nn.ModuleList([
            OptimizedBlock(512, 512),
            OptimizedAttention(512),
            OptimizedBlock(512, 512)
        ])
        
        # Up blocks
        self.up_blocks = nn.ModuleList([
            # First up block (512 -> 512 -> 256)
            nn.Sequential(
                OptimizedBlock(512, 512),
                OptimizedBlock(512, 512),
                OptimizedBlock(512, 512),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(512, 512, kernel_size=3, padding=1)
            ),
            
            # Second up block (512 -> 512 -> 256)
            nn.Sequential(
                OptimizedBlock(512, 512),
                OptimizedBlock(512, 512),
                OptimizedBlock(512, 256),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(256, 256, kernel_size=3, padding=1)
            ),
            
            # Third up block (256 -> 128)
            nn.Sequential(
                OptimizedBlock(256, 256),
                OptimizedBlock(256, 256),
                OptimizedBlock(256, 128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        ])
        
        # Output layers
        self.norm_out = AdaptiveLayerNorm(128)
        self.out = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # Rescale latents
        x = x / 0.18215
        
        # Initial projection
        x = self.init_proj(x)
        
        # Middle blocks
        for block in self.middle_blocks:
            x = block(x)
        
        # Up blocks
        for block in self.up_blocks:
            x = block(x)
        
        # Apply final normalization
        x = self.norm_out(x)
        
        # Output projection
        x = self.out(x)
        
        return x