import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # Number of heads for Query
    n_kv_heads: Optional[int] = None # Number of heads for Key and Value\
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multipler: Optional[float] = None
    norm_eps: float = 1e-5

    # For KV Cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # As written in the paper, the dimension of the embedding must be even.
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameters
    # According to the formula: theta_i = 10000 ^ (-2(i-1)/dim) for i = {1, 2, ... dim/2}
    # Shape: (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (head_dim / 2)
    # formula can be converted to theta_i = 1 / (10000 ^ (2i/dim)) for i = {0, 2, 4, ... dim}
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # Construct the positions ("m")
    # Shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product
    # Shape: (seq_len) outer_product* (head_dim/2) -> (seq_len, head_dim/2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(i * m * theta), where R = 1 as follows:
    # (seq_len, head_dim/2) -> (seq_len, head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (batch, seq_len, h, head_dim) -> (batch, seq_len, h, head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (b, seq_len, h, head_dim/2) * (1, seq_len, 1, head_dim/2) = (batch, seq_len, h, head_dim/2)
    x_rotated = x_complex * freqs_complex
    # (batch, seq_len, h, head_dim/2) -> (batch, seq_len, h, head_dim /2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (batch, seq_len, h, head_dim/2, 2) -> (batch, seq_len, h, head_dim)
    x_out = x_out.flatten(*x.shape)
    return x_out.type_as(x).to(device)

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (batch, seq_len, dim) * (batch, seq_len, 1) = (batch, seq_len, dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        # (dim) * (b, seq_len, dim) = (batch, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return {
            # (batch, seq_len, n_kv_heads, 1, head_dim)
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        }

class SelfAttention(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Indicates the number of heads for the Key and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the heads of Keys and Values should be repeated to match the head of the Queries
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head
        self.head_dim = args.dim // args.n_heads
        
        self.w_q = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape # (batch, seq_len = 1, dim)

        # Apply the W_q, W_k, and W_v matrices to queries, keys and values
        # (batch, 1, dim) -> (batch, 1, h_size_q * head_dim)
        x_q = self.w_q(x)
        # (batch, 1, dim) -> (batch, 1, h_size_kv * head_dim)
        x_k = self.w_k(x)
        x_v = self.w_v(x)

        # (batch, 1, h_q * head_dim) -> (batch, 1, h_q, head_dim)
        x_q = x_q.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (batch, 1, h_kv * head_dim) -> (batch, 1, h_kv, head_dim)
        x_k = x_k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        x_v = x_v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Does not change the shape of the tensors
        x_q = apply_rotary_embeddings(x_q, freqs_complex, device=x.device)
        x_k = apply_rotary_embeddings(x_k, freqs_complex, device=x.device)

        # Repalce the entry in the cache for this token
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = x_k
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = x_v

        # Retrieve all the cached keys and values so far
        # (batch, seq_len_kv, h_kv, head_dim)
        keys = self.cache_k[:batch_size, 0:start_pos+seq_len]
        values = self.cache_v[:batch_size, 0:start_pos+seq_len]

        # Repeat the heads of the Keys and Values to reach the number of heads of the Queries
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (batch, 1, h_q, head_dim) -> (batch, h_q, 1, head_dim)
        x_q = x_q.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (batch, h_q, 1, head_dim) @ (batch, h_q, head_dim, seq_len_kv) -> (batch, h_q, 1, seq_len_kv)
        scores = torch.matmul(x_q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(x_q)

        # (batch, h_q, 1, seq_len) @ (batch, h_q, seq_len_kv, head_dim) -> (batch, h_q, 1, head_dim)
        output = torch.matmul(scores, values)

        # (batch, h_q, 1, head_dim) -> (batch, 1, h_q, head_dim) -> (batch, 1, dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.w_o(output) # (batch, 1, dim) -> (batch, 1, dim)
 
class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedFoward(args)

        # Normalization before the self attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization before the feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (batch, seq_len, dim) + (batch, seq_len, dim) -> (batch, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (batch, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed (KV cache)"

        # (batch, seq_len) -> (batch, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # Retreieve the pairs (m, theta) corresponding to the position [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)
        output = self.output(h).float()
        return output