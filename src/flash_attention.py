import torch
from torch.nn import functional as F
import torch.nn as nn
import math

try:
    from customtriton import FlashAttention
    FLASH_ATTENTION_AVAILABLE = True
    print("Custom FlashAttention is available")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("Custom FlashAttention is not available")

DEBUG_MODE = False

def debug_print(msg, tensors=None):
    if DEBUG_MODE:
        print(f"DEBUG: {msg}")
        if tensors is not None:
            for name, tensor in tensors.items():
                if isinstance(tensor, torch.Tensor):
                    print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}, "
                          f"device={tensor.device}, contiguous={tensor.is_contiguous()}")

class FlashSelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embeddings: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embeddings, d_embeddings * 3, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embeddings, d_embeddings, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embeddings // n_heads

    def forward(self, x, causal_mask=False):
        input_shape = x.shape
        batch_size, sequence_length, d_embeddings = input_shape
        
        if DEBUG_MODE:
            debug_print(f"SelfAttention input: batch_size={batch_size}, seq_len={sequence_length}, dim={d_embeddings}")

        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        use_custom_flash = FLASH_ATTENTION_AVAILABLE
        
        if use_custom_flash:
            try:
                q_flash = q.reshape(batch_size, sequence_length, self.n_heads, self.d_head).permute(0, 2, 1, 3)
                k_flash = k.reshape(batch_size, sequence_length, self.n_heads, self.d_head).permute(0, 2, 1, 3)
                v_flash = v.reshape(batch_size, sequence_length, self.n_heads, self.d_head).permute(0, 2, 1, 3)
                
                q_flash = q_flash.contiguous()
                k_flash = k_flash.contiguous()
                v_flash = v_flash.contiguous()
                
                q_flash_fp32 = q_flash.float()
                k_flash_fp32 = k_flash.float()
                v_flash_fp32 = v_flash.float()
                
                softmax_scale = 1.0 / math.sqrt(self.d_head)
                
                output = FlashAttention.apply(q_flash_fp32, k_flash_fp32, v_flash_fp32, causal_mask, softmax_scale)
                
                if q.dtype != torch.float32:
                    output = output.to(q.dtype)
                
                output = output.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_embeddings)
                
            except Exception as e:
                if DEBUG_MODE:
                    print(f"Custom FlashAttention failed: {str(e)}, falling back to standard attention")
                use_custom_flash = False
        
        if not use_custom_flash:
            q_std = q.view(batch_size, sequence_length, self.n_heads, self.d_head).transpose(1, 2)
            k_std = k.view(batch_size, sequence_length, self.n_heads, self.d_head).transpose(1, 2)
            v_std = v.view(batch_size, sequence_length, self.n_heads, self.d_head).transpose(1, 2)

            attention_scores = (q_std @ k_std.transpose(-2, -1)) / math.sqrt(self.d_head)
            
            if causal_mask:
                mask = torch.ones_like(attention_scores, dtype=torch.bool).triu(1)
                attention_scores = attention_scores.masked_fill(mask, -torch.inf)
            
            attention_probs = F.softmax(attention_scores, dim=-1)
            
            output = attention_probs @ v_std
            
            output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, d_embeddings)

        output = self.out_proj(output)
        
        return output

class FlashCrossAttention(nn.Module):
    def __init__(self, n_heads, d_embeddings: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embeddings, d_embeddings, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embeddings, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embeddings, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embeddings, d_embeddings, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embeddings // n_heads
    
    def forward(self, x, y):
        input_shape = x.shape
        batch_size, sequence_length, d_embeddings = input_shape
        kv_seq_len = y.size(1)
        
        if DEBUG_MODE:
            debug_print(f"CrossAttention input: batch={batch_size}, q_seq={sequence_length}, " 
                       f"kv_seq={kv_seq_len}, dim={d_embeddings}")

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        
        q_std = q.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        k_std = k.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        v_std = v.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        
        attention_scores = (q_std @ k_std.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        output = attention_probs @ v_std
        
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, d_embeddings)

        output = self.out_proj(output)
        
        return output