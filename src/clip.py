import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size, n_embeddings, n_tokens: int): #n_tokens = seq_len
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embeddings)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embeddings))
    
    def forward(self, tokens):
        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_heads: int, n_embeddings: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embeddings)
        self.attention = SelfAttention(n_heads, n_embeddings)
        self.layernorm_2 = nn.LayerNorm(n_embeddings)
        self.linear_1 = nn.Linear(n_embeddings, n_embeddings * 4)
        self.linear_2 = nn.Linear(n_embeddings * 4, n_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch_size, seq_len, dim

        residue = x

        #self attention
        x = self.layernorm_1(x)

        x = self.attention(x, causal_mask=True)

        x += residue

        # feed-forward
        residue = x
        
        x = self.layernorm_2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x) # quick GeLU activation

        x = self.linear_2(x)

        x += residue
        
        return x


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77) #vocab size, embedding, seq_len

        # 12 layers of (num_of_head, embedding)
        self.layers = nn.ModuleList([CLIPLayer(12, 768) for i in range(12)])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long) #convert each tokens (number) into embedding

        # (batch_size, seq_len) -> (batch_size, seq_len, dim(768))
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        
        output = self.layernorm(state)

        return output
