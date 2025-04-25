import torch
import math
from torch.nn import functional as F
import torch.nn as nn
from attention import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # batch_size, channel, (height, width) -> (batch_size, 128, (height, width))
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # batch_size, 128, (height, width) -> batch_size, 128, (height, width)
            VAE_ResidualBlock(128, 128),

            # batch_size, 128, (height, width) -> batch_size, 128, (height, width)
            VAE_ResidualBlock(128, 128),

            # batch_size, 128, (height, width) -> batch_size, 128, (height/2, width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # batch_size, 128, (height/2, width/2) -> batch_size, 256, (height/2, width/2)
            VAE_ResidualBlock(128, 256),

            # batch_size, 256, (height/2, width/2) -> batch_size, 256, (height/2, width/2)
            VAE_ResidualBlock(256, 256),

            # batch_size, 256, (height/2, width/2) -> batch_size, 256, (height/4, width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # batch_size, 256, (height/2, width/2) -> batch_size, 512, (height/4, width/4)
            VAE_ResidualBlock(256, 512),

            # batch_size, 512, (height/4, width/4) -> batch_size, 512, (height/4, width/4)
            VAE_ResidualBlock(512, 512),

            # batch_size, 512, (height/4, width/4) -> batch_size, 512, (height/8, width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            # batch_size, 512, (height/8, width/8) -> batch_size, 512, (height/8, width/8)
            nn.GroupNorm(32, 512),

            # x * sigmoid()
            nn.SiLU(),
            
            # batch_size, 512, (height/8, width/8) -> batch_size, 8, (height/8, width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # batch_size, 8, (height/8, width/8) -> batch_size, 8, (height/8, width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x (batch_size, channel(3), H, W)
        # noise (batch_size, output_channel, H/8, W/8)

        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1)) #(padding left(0), right(1), top(0), bottom(1))
            x = module(x)
        
        # batch_size, 8, H/8, W/8 -> 2 tensors (batch_size, 4, H/8, W/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # batch_size, 4, H/8, W/8 -> batch_size, 4, H/8, W/8
        log_variance = torch.clamp(log_variance, -30, 20) #if the log_variance is too big/small , change it to the defined range (-30, 20)

        # batch_size, 4, H/8, W/8 -> batch_size, 4, H/8, W/8
        variance = log_variance.exp() # convert log_variance to variance

        # batch_size, 4, H/8, W/8 -> batch_size, 4, H/8, W/8
        standard_deviation = variance.sqrt()

        # Z = N(0, 1) -> N(mean, variance) = X => X = mean + standard_deviation  * Z
        x = mean + standard_deviation * noise

        x *= 0.18215

        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=8192, embedding_dim=4, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        b, c, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, c)
        
        distances = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
                   torch.sum(self.embeddings.weight ** 2, dim=1) - \
                   2 * torch.matmul(z_flat, self.embeddings.weight.t())
                   
        min_indices = torch.argmin(distances, dim=1)
        z_q = self.embeddings(min_indices).reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        commitment_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                          torch.mean((z_q - z.detach()) ** 2)
        
        z_q = z + (z_q - z).detach()
        
        return z_q, commitment_loss, min_indices

class VAE_Encoder_Optimized(nn.Module):
    def __init__(self):
        super().__init__()
        
        from attention import OptimizedBlock, OptimizedAttention, AdaptiveLayerNorm
        
        # Initial convolution
        self.init_conv = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        
        # Down blocks
        self.down_blocks = nn.ModuleList([
            # First block (128 -> 128)
            nn.Sequential(
                OptimizedBlock(128, 128),
                OptimizedBlock(128, 128),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
            ),
            
            # Second block (128 -> 256)
            nn.Sequential(
                OptimizedBlock(128, 256),
                OptimizedBlock(256, 256),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
            ),
            
            # Third block (256 -> 512)
            nn.Sequential(
                OptimizedBlock(256, 512),
                OptimizedBlock(512, 512),
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
            )
        ])
        
        # Middle blocks
        self.middle_blocks = nn.ModuleList([
            OptimizedBlock(512, 512),
            OptimizedAttention(512),
            OptimizedBlock(512, 512)
        ])
        
        # Output layers
        self.norm_out = AdaptiveLayerNorm(512)
        self.out = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1)
        )
        
        # Vector quantizer
        self.vector_quantizer = VectorQuantizer(num_embeddings=8192, embedding_dim=4, beta=0.25)
        
    def forward(self, x, noise):
        # Initial convolution
        x = self.init_conv(x)
        
        # Down blocks
        for block in self.down_blocks:
            x = block(x)
        
        # Middle blocks
        for block in self.middle_blocks:
            x = block(x)
        
        # Apply final normalization
        x = self.norm_out(x)
        
        # Output projection
        x = self.out(x)
        
        # Split into mean and log variance
        mean, log_variance = torch.chunk(x, 2, dim=1)
        
        # Clamp log variance for numerical stability
        log_variance = torch.clamp(log_variance, -30, 20)
        
        # Calculate standard deviation
        variance = torch.exp(log_variance)
        std = torch.sqrt(variance)
        
        # Reparameterize
        x = mean + std * noise
        x = x * 0.18215
        
        # Apply vector quantization if needed
        # x, commitment_loss, indices = self.vector_quantizer(x)
        
        return x
