import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding():
    pass

class CLIPLayer():
    pass

class CLIP(nn.Module):
    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77) #vocab size, embedding, seq_len

        # 12 layers of (num_of_head, embedding)
        self.layers = nn.Module([CLIPLayer(12, 768) for i in range(12)])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long) #convert each tokens (number) into embedding

        # (batch_size, seq_len) -> (batch_size, seq_len, dim(768))
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        
        output = self.layernorm(state)

        return output
