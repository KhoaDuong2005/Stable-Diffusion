import torch
import torch.nn as nn
from torch.nn import functional as F

class UNET_ResidualBlock(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, n_time=1280):
        super().__init__()
        self.groupnorm_feature= nn.GroupNorm(32, input_channel)
        self.conv_feature = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, output_channel)

        self.groupnorm_merged = nn.GroupNorm(32, output_channel)
        self.conv_merged = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)

        if input_channel == output_channel:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(input_channel, output_channel, kernel_size=1, padding=0)

    def forward(self, feature, time):
        # feature = batch_size, in_channel, H, W
        # time = 1, 1280

        residue = feature

        feature = self.groupnorm_feature(feature)

        feature = F.silu(feature)

        feature = self.conv_feature(feature)

        time = F.silu(time)

        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)

DEBUG_MODE = True

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_heads: int, n_embeddings: int, d_prompt=768, attention_type="xformers"):
        super().__init__()
        channels = n_heads * n_embeddings

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.layernorm_3 = nn.LayerNorm(channels)
        
        # Choose attention implementation based on the parameter
        if attention_type == "flashattention":
            if DEBUG_MODE:
                print("Using flash attention for UNET")
            from flash_attention import FlashSelfAttention, FlashCrossAttention
            self.attention_1 = FlashSelfAttention(n_heads, channels, in_proj_bias=False)
            self.attention_2 = FlashCrossAttention(n_heads, channels, d_prompt, in_proj_bias=False)
        elif attention_type == "xformers":
            if DEBUG_MODE:
                print("Using xformers attention for UNET")
            from attention import SelfAttention, CrossAttention
            self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias=False)
            self.attention_2 = CrossAttention(n_heads, channels, d_prompt, in_proj_bias=False)
        elif attention_type == "default":
            print("Using default attention for UNET")
            from attention import SelfAttention, CrossAttention
            self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias=False)
            self.attention_2 = CrossAttention(n_heads, channels, d_prompt, in_proj_bias=False)
            
        else:
            raise ValueError(f"Unknown attention type: {attention_type}. Must be 'flashattention' or 'xformers'")

        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(channels * 4, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, prompt):
        # x = batch_size, features, H, W
        # prompt = batch_size, seq_len, dim

        residue_long = x
        
        x = self.groupnorm(x)

        x = self.conv_input(x)

        n, c, h, w = x.shape

        # batch_size, features, H, W -> batch_size, features, H * W
        x = x.view((n, c, h * w))

        #batch_size, features, H * W -> batch_size, H * W, features
        x = x.transpose(-2, -1)

        # normalization + self attention with skip connection

        residue_short = x
        
        x = self.layernorm_1(x)
        x = self.attention_1(x)

        x += residue_short

        residue_short = x

        # normalization + cross attention with skip connection
        x = self.layernorm_2(x)

        x = self.attention_2(x, prompt)

        x += residue_short

        residue_short = x

        # normalization + feed forward with GeLU and skip connection
        x = self.layernorm_3(x)

        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)

        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)

        x += residue_short

        # batch_size, H * W, features -> batch_size, features, H * W
        x = x.transpose(-2, -1)

        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
        

class TimeEmbedding(nn.Module):
    def __init__(self, n_embeddings):
        super().__init__()
        self.linear_1 = nn.Linear(n_embeddings, n_embeddings * 4)
        self.linear_2 = nn.Linear(n_embeddings * 4, n_embeddings * 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1, 320)
         x = self.linear_1(x)
         
         x = F.silu(x)

         x = self.linear_2(x)

        # (1, 1280)
         return x



class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, prompt: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock): # if layer is UNET_AttentionBlock, then pass prompt in the layer to get the "attention" output
                x = layer(x, prompt)

            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)

            else:
                x = layer(x)

        return x




class UNET(nn.Module):
    def __init__(self, attention_type="xformers"):
        super().__init__()
        self.attention_type = attention_type
        self.encoders = nn.ModuleList([
            # batch_size, 4, H/8, W/8
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40, attention_type=attention_type)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40, attention_type=attention_type)),

            # -------------------------------------------------------------------------#
            # batch_size, 320, H/8, W/8 -> # batch_size, 320, H/16, W/16
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            #decrease the size
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80, attention_type=attention_type)),

            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80, attention_type=attention_type)),

            # -------------------------------------------------------------------------#
            # batch_size, 640, H/16, W/16 -> batch_size, 640, H/32, W/32
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160, attention_type=attention_type)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160, attention_type=attention_type)),

            # -------------------------------------------------------------------------#
            # batch_size, 1280, H/32, W/32 -> batch_size, 1280, H/64, W/64
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(

            UNET_ResidualBlock(1280, 1280),

            UNET_AttentionBlock(8, 160, attention_type=attention_type),

            UNET_ResidualBlock(1280, 1280), 
        )

        self.decoders = nn.ModuleList([
            # batch_size, 2560, H/64, W/64 -> batch_size, 1280, H/64, W/64
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160, attention_type=attention_type)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160, attention_type=attention_type)),

            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160, attention_type=attention_type), Upsample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80, attention_type=attention_type)),
        
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80, attention_type=attention_type)),

            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80, attention_type=attention_type), Upsample(640)),

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40, attention_type=attention_type)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40, attention_type=attention_type)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40, attention_type=attention_type)),
        ])

    def forward(self, x, prompt, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # prompt: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, prompt, time)
            skip_connections.append(x)

        x = self.bottleneck(x, prompt, time)

        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, prompt, time)

        
        return x

class UNET_Output_Layer(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, input_channel)
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)

    def forward(self, x):
        # x = batch_size, 320, H/8, W/8

        x = self.groupnorm(x)

        x = F.silu(x)

        x = self.conv(x)

        # batch_size, 4,  H/8/ W/8
        return x


class Diffusion(nn.Module): 
    def __init__(self, attention_type="xformers"):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET(attention_type=attention_type)
        self.final = UNET_Output_Layer(320, 4)

    def forward(self, latent: torch.Tensor, prompt: torch.Tensor, time: torch.Tensor):
        # ------------ INPUT -----------------
        # latent = (batch_size, 4, H/8, W/8)
        # prompt = (batch_size, seq_len, dim) (dim = 768)
        # time = (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (batch_size, 4, H/8, W/8) -> (batch_size, 320, H/8, W/8)
        output = self.unet(latent, prompt, time)

        # (batch_size, 320, H/8, W/8) -> (batch_size, 4, H/8, W/8)
        output = self.final(output)

        return output
