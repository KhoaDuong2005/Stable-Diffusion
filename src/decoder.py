import torch
from torch.nn import functional as F
import torch.nn as nn
from attention import SelfAttention, DiTBlock, VAE_ResidualBlock, VAE_AttentionBlock, AdaptiveLayerNorm



class VA_Decoder(nn.Module):
    def __init__(self, use_vq=True, use_dit=False, use_adaptive_norm=False):
        super().__init__()
        self.use_vq = use_vq
        
        self.decoder_layers = nn.ModuleList([
            # (batch_size, 4, H/8, W/8) -> (batch_size, 4, H/8, W/8)
            nn.Conv2d(4, 4, kernel_size=1, padding=0), 
            
            # (batch_size, 4, H/8, W/8) -> (batch_size, 512, H/8, W/8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(512, 512, use_adaptive_norm=use_adaptive_norm),
            
            VAE_AttentionBlock(512, use_dit=use_dit), # (batch_size, 512, H/8, W/8) -> (batch_size, 512, H/8, W/8)

            VAE_ResidualBlock(512, 512, use_adaptive_norm=use_adaptive_norm),
            VAE_ResidualBlock(512, 512, use_adaptive_norm=use_adaptive_norm),
            VAE_ResidualBlock(512, 512, use_adaptive_norm=use_adaptive_norm),
            VAE_ResidualBlock(512, 512, use_adaptive_norm=use_adaptive_norm),
            
            nn.Upsample(scale_factor=2, mode="nearest"), # (batch_size, 512, H/8, W/8) -> (batch_size, 512, H/4, W/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # (batch_size, 512, H/4, W/4) -> (batch_size, 512, H/4, W/4)
            
            VAE_ResidualBlock(512, 512, use_adaptive_norm=use_adaptive_norm),
            VAE_ResidualBlock(512, 512, use_adaptive_norm=use_adaptive_norm),
            VAE_ResidualBlock(512, 512, use_adaptive_norm=use_adaptive_norm),
            
            nn.Upsample(scale_factor=2, mode="nearest"), # (batch_size, 512, H/4, W/4) -> (batch_size, 512, H/2, W/2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # (batch_size, 512, H/2, W/2) -> (batch_size, 512, H/2, W/2)
            
            VAE_ResidualBlock(512, 256, use_adaptive_norm=use_adaptive_norm), # (batch_size, 512, H/2, W/2) -> (batch_size, 256, H/2, W/2)
            VAE_ResidualBlock(256, 256, use_adaptive_norm=use_adaptive_norm),
            VAE_ResidualBlock(256, 256, use_adaptive_norm=use_adaptive_norm),
            
            nn.Upsample(scale_factor=2, mode="nearest"), # (batch_size, 256, H/2, W/2) -> (batch_size, 256, H, W)
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # (batch_size, 256, H, W) -> (batch_size, 256, H, W)
            
            VAE_ResidualBlock(256, 128, use_adaptive_norm=use_adaptive_norm), # (batch_size, 256, H, W) -> (batch_size, 128, H, W)
            VAE_ResidualBlock(128, 128, use_adaptive_norm=use_adaptive_norm),
            VAE_ResidualBlock(128, 128, use_adaptive_norm=use_adaptive_norm),
        ])
        
        if use_adaptive_norm:
            self.final_norm = AdaptiveLayerNorm(128, num_embeddings=1000)
            
            import math
            # convert to 2D after adaptive norm
            self.to_2d = lambda x: x.transpose(1, 2).reshape(
                x.size(0), 128, int(math.sqrt(x.size(1))), int(math.sqrt(x.size(1)))
            )
        
        else:
            self.final_norm = nn.GroupNorm(32, 128)
            self.to_2d = lambda x: x
        
        self.final_layers = nn.ModuleList([
            nn.SiLU(), # x * sigmoid()
            nn.Conv2d(128, 3, kernel_size=3, padding=1), # (batch_size, 128, H, W) -> (batch_size, 3, H, W)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (batch_size, 4, H/8, W/8)
        x /= 0.18215
        
        for module in self.decoder_layers:
            x = module(x)
        
        if isinstance(self.final_norm, AdaptiveLayerNorm):
            b, c, h, w = x.shape
            x_flat = x.flatten(2).transpose(1, 2) # (b, c, h, w) -> (b, h * w, c)
            x_flat = self.final_norm(x_flat)
            x_flat = x_flat.transpose(1, 2).reshape(b, c, h, w) # (b, h * w, c) -> (b, c, h, w)
        else:
            x = self.final_norm(x)
            
        for module in self.final_layers:
            x = module(x)
        
        return x # (batch_size, 3, H, W)
            
         
         
"""                   
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
        

"""  
