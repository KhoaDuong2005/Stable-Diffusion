import torch
from torch.nn import functional as F
import torch.nn as nn
from attention import self_attention_heads

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
            self.residual_layer = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=0)

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
        super.__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (batch_size, features, H, W)
        residue = x

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