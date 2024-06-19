import einops
# from fancy_einsum import einsum
from dataclasses import dataclass
# from easy_transformer import EasyTransformer
import torch
import torch.nn as nn
# import numpy as np
# import math
# from easy_transformer.utils import get_corner, gelu_new, tokenize_and_concatenate
# import tqdm.auto as tqdm


# Layer Normalization
"""
Make mean zero
Normalize to have variance=1
Scale with learned weights
Translate with learned bias"""


@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    
cfg = Config()

class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))
        
    def forward(self, residual):
        if cfg.debug: print("Residual:", residual.shape)
        #Make mean zero
        residual = residual - einops.reduce(residual, "batch position d_model -> batch position 1", "mean") 
        scale = (einops.reduce(residual.pow(2), "batch position d_model -> batch position 1", "mean") + cfg.layer_norm_eps).sqrt()
        normalized = residual / scale
        normalized = normalized * self.w + self.b
        if cfg.debug: print("Normalized:", residual.shape)
        return normalized
        


# Embedding
# Positional Embedding
# Attention
# MLP
# Transformer Block
# Unembedding
# Full Transformer
# Tests


