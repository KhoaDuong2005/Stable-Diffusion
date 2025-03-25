import torch
from torch.nn import functional as F
import torch.nn as nn
import math



class Attention(nn.Module):
    def __init__(self, n_heads: int, d_embeddings: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embeddings, d_embeddings * 3, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embeddings, d_embeddings, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embeddings // n_heads 

    def forward(self, x, casual_mask=False):
        batch_size, sequence_length, d_embeddings  = x.shape

        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1) #query, key, value

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            # upper triangular
            mask = torch.ones_like(weight, dtype=torch.bool).triu(x)

            weight.masked_fill(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1, 2)
        
        output = self.out_proj(output)

        return output



