import torch
from torch.nn import functional as F
import torch.nn as nn
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super.__init__(
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
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
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
