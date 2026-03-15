import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class TuringLLMConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 1024
    hidden_size: int = 4096
    norm_eps: float = 1e-5


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale


class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
    def forward(self, x, input_pos=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.n_embd, config.hidden_size * 2)
        self.down_proj = nn.Linear(config.hidden_size, config.n_embd)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)
        
    def forward(self, x, input_pos):
        attn_x = self.attn(self.norm_1(x), input_pos)
        x = x + attn_x
        mlp_down_x = self.mlp(self.norm_2(x))
        x = x + mlp_down_x
        return attn_x, mlp_down_x, x


class TuringLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = TuringLLMConfig()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe=nn.Embedding(self.config.block_size, self.config.n_embd),
            h=nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
            norm_f=RMSNorm(self.config.n_embd, self.config.norm_eps),
        ))
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
    def forward(self, idx, targets=None, input_pos=None, return_all_logits=False):
        # B (Batch), T (Tokens)
        _, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence length {T}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            _, _, x = block(x, input_pos)

        if targets is not None:
            x = self.transformer.norm_f(x)
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        if return_all_logits:
            x = self.transformer.norm_f(x)
        else:
            x = self.transformer.norm_f(x[:, -1:, :])
            
        logits = self.lm_head(x)
        return logits, None

