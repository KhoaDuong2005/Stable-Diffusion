import torch
from torch.nn import functional as F
import torch.nn as nn
from attention import SelfAttention

class VAE_ResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, input_channel)
        self.conv_1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, output_channel)
        self.conv_2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)

        if input_channel == output_channel:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(input_channel, output_channel, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (batch_size, input_channels, H, W)
        
        residue = x
        
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        x = x + self.residual_layer(residue)

        return x



class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (batch_size, features, H, W)
        residue = x

        x = self.groupnorm(x)

        n, c, h, w = x.shape # n = batch_size | c = channel | h = height | w = width

        # batch_size, features, H, W -> batch_size, features, H * W
        x = x.view(n, c, h * w)

        # batch_size, features, H * W -> batch_size, H * W, features
        x = x.transpose(-1, -2)

        # transpose to do attention and then transpose back
        x = self.attention(x)

        #batch_size, H * W, features -> batch_size, features, H * W
        x = x.transpose(-1, -2)
         
        # batch_size, features, H * W -> batch_size, features, H, W
        x = x.view((n, c, h, w))

        x += residue
        
        return x	


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