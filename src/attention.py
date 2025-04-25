import torch
from torch.nn import functional as F
import torch.nn as nn
import math

try:
    import xformers.ops
    from xformers.ops.fmha.attn_bias import LowerTriangularMask
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    print("xformers is not available")

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embeddings, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embeddings, d_embeddings * 3, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embeddings, d_embeddings, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embeddings // n_heads

    def forward(self, x, causal_mask=False):
        batch_size, sequence_length, d_embeddings = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        if XFORMERS_AVAILABLE:
            q = q.view(batch_size, sequence_length, self.n_heads, self.d_head)
            k = k.view(batch_size, sequence_length, self.n_heads, self.d_head)
            v = v.view(batch_size, sequence_length, self.n_heads, self.d_head)
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            attention_bias = LowerTriangularMask() if causal_mask else None
            output = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attention_bias)
            output = output.view(batch_size, sequence_length, d_embeddings)
        else:
            interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
            q = q.view(interim_shape).transpose(1, 2)
            k = k.view(interim_shape).transpose(1, 2)
            v = v.view(interim_shape).transpose(1, 2)
            weight = q @ k.transpose(-2, -1)
            
            if causal_mask:
                mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
                weight = weight.masked_fill(mask, -torch.inf)
                
            weight /= math.sqrt(self.d_head)
            weight = F.softmax(weight, dim=-1)
            output = weight @ v
            output = output.transpose(1, 2).contiguous()
            output = output.view(batch_size, sequence_length, d_embeddings)
        
        output = self.out_proj(output)
        return output

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim, num_embeddings=None):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.embedding = nn.Embedding(num_embeddings, dim * 2) if num_embeddings else None
        self.dim = dim

    def forward(self, x, emb_idx=None):
        orig_shape = x.shape
        need_reshape = len(orig_shape) > 3
        
        if need_reshape:
            b, c, h, w = x.shape
            x = x.reshape(b, c, h*w).permute(0, 2, 1)  # (b, c, h, w) -> (b, h*w, c)
            
        x = self.layer_norm(x)
        
        if self.embedding is not None and emb_idx is not None:
            scale, shift = self.embedding(emb_idx).chunk(2, dim=-1)
            x = x * (1 + scale) + shift
            
        if need_reshape:
            x = x.permute(0, 2, 1).reshape(b, c, h, w)  # (b, h*w, c) -> (b, c, h, w)
            
        return x

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        residue = x
        n, c, h, w = x.shape
        
        x = self.groupnorm(x)
        x = x.reshape(n, c, h * w).permute(0, 2, 1)  # (n, c, h*w) -> (n, h*w, c)
        x = self.attention(x)
        x = x.permute(0, 2, 1).reshape(n, c, h, w)  # (n, h*w, c) -> (n, c, h, w)
        
        return x + residue

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(heads, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residue = x
        
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)

class OptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.norm(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip(x)

class OptimizedAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.num_heads = num_heads
        self.channels_per_head = channels // num_heads

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.channels_per_head, h * w)
        q, k, v = qkv.unbind(dim=1)
        
        if XFORMERS_AVAILABLE:
            q = q.reshape(b * self.num_heads, self.channels_per_head, h * w).transpose(1, 2).contiguous()
            k = k.reshape(b * self.num_heads, self.channels_per_head, h * w).transpose(1, 2).contiguous()
            v = v.reshape(b * self.num_heads, self.channels_per_head, h * w).transpose(1, 2).contiguous()
            
            # Use half precision for xformers
            dtype = torch.float16 if x.device.type == 'cuda' else torch.float32
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
            
            out = xformers.ops.memory_efficient_attention(q, k, v)
            out = out.to(x.dtype)  # Convert back to original dtype
            out = out.transpose(1, 2).reshape(b, c, h, w)
        else:
            q = q.reshape(b, self.num_heads, self.channels_per_head, h * w).transpose(2, 3)
            k = k.reshape(b, self.num_heads, self.channels_per_head, h * w).transpose(2, 3)
            v = v.reshape(b, self.num_heads, self.channels_per_head, h * w).transpose(2, 3)
            
            attention = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.channels_per_head), dim=-1)
            out = (attention @ v).transpose(2, 3).reshape(b, c, h, w)
            
        out = self.proj(out)
        return out + residual
    
    
class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embeddings: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embeddings, d_embeddings, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embeddings, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embeddings, bias=in_proj_bias)

        self.out_proj = nn.Linear(d_embeddings, d_embeddings, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embeddings // n_heads
    
    def forward(self, x, y):
        # x = (latent) = (batch_size, seq_len_Q, dim_Q)
        # y = (prompt) = (batch_size, seq_len_KV, dim_KQ) = batch_size, 77, 768
        input_shape = x.shape
        batch_size, sequence_length, d_embeddings = input_shape

        
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        if XFORMERS_AVAILABLE:
            q = q.view(batch_size, -1, self.n_heads, self.d_head)
            k = k.view(batch_size, -1, self.n_heads, self.d_head)
            v = v.view(batch_size, -1, self.n_heads, self.d_head)

            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

            output = xformers.ops.memory_efficient_attention(
                q, k, v
            )

        else:
            interim_shape = (batch_size, -1, self.n_heads, self.d_head)

            q = q.view(interim_shape).transpose(1, 2)
            k = k.view(interim_shape).transpose(1, 2)
            v = v.view(interim_shape).transpose(1, 2)

            weight = q @ k.transpose(-2, -1)

            weight /=   math.sqrt(self.d_head)

            weight = F.softmax(weight, dim=-1)

            output = weight @ v

            output = output.transpose(1, 2).contiguous()

        output = output.view(batch_size, sequence_length, d_embeddings)

        output = self.out_proj(output)

        return output
