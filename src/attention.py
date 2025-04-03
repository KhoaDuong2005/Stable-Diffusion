import torch
from torch.nn import functional as F
import torch.nn as nn
import math

try:
    import xformers.ops
    XFORMERS_AVAILABLE = True
    print("xformers is available")
except ImportError:
    XFORMERS_AVAILABLE = False
    print("xformers is not available")


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embeddings: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embeddings, d_embeddings * 3, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embeddings, d_embeddings, bias=out_proj_bias)

        self.n_heads = n_heads #number of head
        self.d_head = d_embeddings // n_heads #size of each head (dimension)

    def forward(self, x, casual_mask=False):
        input_shape = x.shape
        batch_size, sequence_length, d_embeddings  = input_shape

        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # batch_size, seq_len, dim -> batchsize, seq_len, dim/3 -> 3 tensors of shape batch_size, seq_len, dim
        q, k, v = self.in_proj(x).chunk(3, dim=-1) #query, key, value

        #batch_size, seq_len, dim -> (batch_size, seq_len, H, dim / H), and with transpose 1,2 => (batch_size, H, seq_len, dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        attention_mask = None

        if casual_mask:
            attention_mask = torch.ones(
                (batch_size, self.n_heads, sequence_length, sequence_length),
                dtype=torch.bool, 
                device=x.device
            )

            attention_mask = torch.triu(attention_mask, diagonal=1)

        # if xformers is available, use it
        if XFORMERS_AVAILABLE:
            attention_bias = None
            if casual_mask:
                attention_bias = torch.zeros_like(attention_mask, dtype=torch.float)
                attention_bias = attention_bias.masked_fill(attention_mask, float("-inf"))

            # apply xformers attention
            output = xformers.ops.memory_efficient_attention(
                q, k, v,
                attn_bias=attention_bias,
                scale=1.0 / math.sqrt(self.d_head),
            )

        #if not xformers, use the standard attention
        else:
            weight = q @ k.transpose(-2, -1)

            if casual_mask:
            # upper triangular is made up of 1
                mask = torch.ones_like(weight, dtype=torch.bool).triu(1)

                weight = weight.masked_fill(mask, -torch.inf)

            weight /= math.sqrt(self.d_head)

            weight = F.softmax(weight, dim=-1)

            # (batch_size, H, seq_len, seq_len) @ (batch_size, H, seq_len, dim / H) - > (batch_size, H, seq_len, dim / H)
            output = weight @ v
        


        # (batch_size, H, seq_len, dim / H) -> (batch_size, seq_len, H, dim / H)
        output = output.transpose(1, 2)

        # (batch_size, seq_len, H, dim / H) -> (batch_size, seq_len, dim)
        output = output.reshape(input_shape)
        
        output = self.out_proj(output)

        return output


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

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        if XFORMERS_AVAILABLE:
            # apply xformers attention
            output = xformers.ops.memory_efficient_attention(
                q, k, v,
                attn_bias=None,
                scale=1.0 / math.sqrt(self.d_head),
            )
        
        # if xformers is not available, use the standard attention
        else:
            weight = q @ k.transpose(-2, -1)

            weight /=   math.sqrt(self.d_head)

            weight = F.softmax(weight, dim=-1)

            output = weight @ v


        output = output.transpose(1, 2).contiguous()

        output = output.view(input_shape)

        output = self.out_proj(output)

        return output


