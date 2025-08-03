import torch
import torch.nn as nn
import json
import os, pprint
import matplotlib.pyplot as plt

os.getcwd()

with open("config.json", "r") as f:
    GPT_CONFIG_124M = json.load(f)
    pprint.pprint(GPT_CONFIG_124M)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim:int, eps:float = 1e-5):
        """ Layer normalization module.
        Args:
            normalized_shape (int or tuple): Input shape from an expected input of size.
            eps (float, optional): A value added to the denominator for numerical stability. Default: 1e-5
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        """ Forward pass of the layer normalization.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim).
        Returns:
            torch.Tensor: Layer normalized tensor.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.std(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        """ Forward pass of the GELU activation function.
        
        This implements the Gaussian Error Linear Unit (GELU) activation function:
        GELU(x) = 0.5x * (1 + tanh(sqrt(2/π) * (x + 0.044715x^3)))
        
        Terms breakdown:
        - 0.5x: Scales the output
        - sqrt(2/π): Normalizing constant ≈ 0.7979
        - 0.044715x^3: Cubic term that helps approximate the Gaussian CDF
        - tanh(): Squashes values to range (-1,1)
        - (1 + tanh()): Shifts the range to (0,2)
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output after GELU activation
        """
        """ Forward pass of the GELU activation function."""
        A = torch.sqrt(torch.tensor(2 / torch.pi))
        B = (x + 0.044715 * torch.pow(x, 3))
        return .5 * x * (1 + torch.tanh(A * B ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb_dim = cfg['emb_dim']
        self.layers = nn.Sequential(
            nn.Linear(self.emb_dim, 4 * self.emb_dim),
            GELU(),
            nn.Linear(4 * self.emb_dim, self.emb_dim),
        )
    
    def forward(self, x):
        """ Forward pass of the feed"""
        return self.layers(x)

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)  
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], float('-inf')
        )
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            )
            for _ in range(num_heads)
        ])
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.attn = MultiHeadAttention(
            d_in=self.cfg['emb_dim'],
            d_out=self.cfg['emb_dim'],
            context_length=self.cfg['context_length'],
            dropout=self.cfg['drop_rate'],
            num_heads=self.cfg['n_heads'],
            dropout=self.cfg['drop_rate'],
            qkv_bias=self.cfg['qkv_bias']
        )
        self.ff = FeedForward(self.cfg)
        self.norm1 = LayerNorm(self.cfg['emb_dim'])
        self.norm2 = LayerNorm(self.cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(self.cfg['drop_rate'])
        
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x) 
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(self.cfg['vocab_size'], self.cfg['emb_dim'])
        self.pos_emb = nn.Embedding(self.cfg['context_length'], self.cfg['emb_dim'])
        self.drop_emb = nn.Dropout(self.cfg['drop_rate'])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(self.cfg) for _ in range(self.cfg['n_layers'])]
        )
        
        self.final_norm = LayerNorm(self.cfg['emb_dim'])
        
        self.out_head = nn.Linear(
            self.cfg['emb_dim'], self.cfg['vocab_size'], bias=False
        )
    
    def forward(self, x):
        """Forward pass of the GPT model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size).
        """
        return x


torch.manual_seed(123)
        
gptmodel = self = GPTModel(GPT_CONFIG_124M)


dummy_input = torch.tensor([[1, 5, 2], [4, 3, 0]])  # shape: (batch_size=2, seq_len=3)
self.tok_emb(dummy_input).shape

gelu, relu = GELU(), nn.ReLU()
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
plt.show()


ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)