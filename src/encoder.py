import torch
from torch.nn import functional as F
import torch.nn as nn

from attention import SelfAttention, VAE_AttentionBlock, VAE_ResidualBlock, AdaptiveLayerNorm


    
class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
        
    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1), mode="replicate") # Padding to maintain spatial dimensions
        return self.conv(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=8192, embedding_dim=4, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        # z (batch_size, embedding_dim, H, W)
        b, d, h, w = z.shape
        assert d == self.embedding_dim, f"Expected embedding dimension {self.embedding_dim}, but got {d}"

        # flatten z to find the nearest embedding
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, d)
        
        # calculate distances to embeddings vectors
        d_matrix = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embeddings.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flat, self.embeddings.weight.t())
            
        min_encoding_indices = torch.argmin(d_matrix, dim=1)
        z_q = self.embeddings(min_encoding_indices).reshape(b, h, w, d).permute(0, 3, 1, 2) # (b, h, w, d) -> (b, d, h, w)
        
        commitment_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
            torch.mean((z_q - z.detach()) ** 2)
        
        z_q = z + (z_q - z).detach() # straight-through estimator for backpropagation
        
        return z_q, commitment_loss, min_encoding_indices
    
class VAE_Encoder(nn.Module):
    def __init__(self, use_vq=True, use_dit=False, use_adaptive_norm=False):
        super().__init__()
        self.use_vq = use_vq
    
        
        self.encoder_layers = nn.ModuleList([
            #initial convolution (batch_size, channel, (height, width) -> (batch_size, 128, (height, width))
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            # batch_size, 128, (height, width) -> batch_size, 128, (height, width)
            VAE_ResidualBlock(128, 128, use_adaptive_norm=use_adaptive_norm),
            VAE_ResidualBlock(128, 128, use_adaptive_norm=use_adaptive_norm),
            
            DownSampleBlock(128, 128), # batch_size, 128, (height, width) -> batch_size, 128, (height/2, width/2)
            
            # batch_size, 128, (height/2, width/2) -> batch_size, 256, (height/2, width/2)
            VAE_ResidualBlock(128, 256, use_adaptive_norm=use_adaptive_norm),
            VAE_ResidualBlock(256, 256, use_adaptive_norm=use_adaptive_norm),
            
            DownSampleBlock(256, 256), # batch_size, 256, (height/2, width/2) -> batch_size, 256, (height/4, width/4)
            
            # batch_size, 256, (height/4, width/4) -> batch_size, 512, (height/4, width/4)
            VAE_ResidualBlock(256, 512, use_adaptive_norm=use_adaptive_norm),
            VAE_ResidualBlock(512, 512, use_adaptive_norm=use_adaptive_norm),
            
            DownSampleBlock(512, 512), # batch_size, 512, (height/4, width/4) -> batch_size, 512, (height/8, width/8)
            
            VAE_ResidualBlock(512, 512, use_adaptive_norm=use_adaptive_norm),
            VAE_ResidualBlock(512, 512, use_adaptive_norm=use_adaptive_norm),
            VAE_ResidualBlock(512, 512, use_adaptive_norm=use_adaptive_norm),
            
            VAE_AttentionBlock(512, use_dit=use_dit), # batch_size, 512, (height/8, width/8) -> batch_size, 512, (height/8, width/8)
            
            VAE_ResidualBlock(512, 512, use_adaptive_norm=use_adaptive_norm),
        ])
            
        if use_adaptive_norm:
            self.final_norm = AdaptiveLayerNorm(512, num_embeddings=1000)
            
            import math
            #convert to 2D after adaptive norm
            self.to_2d = lambda x: x.transpose(1, 2).reshape(x.size(0), 512, int(math.sqrt(x.size(1))), int(math.sqrt(x.size(1))))
            
        else:
            self.final_norm = nn.GroupNorm(32, 512)
            self.to_2d = lambda x: x
            

        self.final_layers = nn.ModuleList([
            nn.SiLU(), # x * sigmoid()
            nn.Conv2d(512, 8, kernel_size=3, padding=1), # batch_size, 512, (height/8, width/8) -> batch_size, 8, (height/8, width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0), # batch_size, 8, (height/8, width/8) -> batch_size, 8, (height/8, width/8)
        ])
        
        self.vector_quantizer = VectorQuantizer(num_embeddings=8192, embedding_dim=4) if use_vq else None

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x (batch_size, channel(3), H, W)
        # noise (batch_size, output_channel, H/8, W/8)

        for module in self.encoder_layers:
            x = module(x)
            
        if isinstance(self.final_norm, AdaptiveLayerNorm):
            b, c, h, w = x.shape
            x_flat = x.flatten(2).transpose(1, 2) # (b, c, h, w) -> (b, h * w, c)
            x_flat = self.final_norm(x_flat)
            x = x_flat.transpose(1, 2).reshape(b, c, h, w) # (b, h * w, c) -> (b, c, h, w)
        else:
            x = self.final_norm(x)
        
        for module in self.final_layers:
            x = module(x)
            
        # split into mean and log variance
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        
        # sample from distribution
        x = mean + stdev * noise
        
        # apply vq if enabled
        if self.use_vq and self.vector_quantizer:
            x, vq_loss, idx = self.vector_quantizer(x)
        else:
            vq_loss = None
        
        x = x * 0.18215 #0.13025

        if self.use_vq and vq_loss is not None:
            return x, vq_loss

        return x

        
"""
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

"""
