import math
import numpy as np
import lib.Solvers as Solvers
import torch
from torch.nn import functional as F
import torch.nn as nn
import reprlib
import sys


class COALA_Linear(nn.Module):
    def __init__(self, 
                ratio: int, 
                compress_strategy="empty", 
                fp16=False,
                params=None,
                base_tensor: torch.Tensor = None,
                X: torch.Tensor = None,
            ):


        super().__init__()
        m, n = base_tensor.shape
        r = math.ceil(ratio * m * n / (m + n))
        r = min(r, m, n)
        r = max(r, 1)
        self.r = r
        assert self.r >= 1

        dtype = torch.float32
        if fp16:
            dtype = torch.float16
            
        self.adapter_A = nn.Parameter(base_tensor.new_empty(n, r, dtype=dtype))
        self.adapter_B = nn.Parameter(base_tensor.new_empty(r, m, dtype=dtype))

        self.compress_strategy = compress_strategy
        self.params = params
        self.X = X

        self.set_parameters(base_tensor)

    @torch.no_grad()
    def set_parameters(self, base_tensor):
        if self.compress_strategy == "empty":
            torch.nn.init.zeros_(self.adapter_B)
            torch.nn.init.zeros_(self.adapter_A)
            return
        
        if self.X != None:
            self.X = self.X.to(torch.device('cuda')).to(torch.float32)
        
        if self.compress_strategy == "svd":
            B, A = Solvers.svd_low_rank_approx(base_tensor, self.r)
            
        elif self.compress_strategy == "coala":
            
            mu = self.params['mu']
            if self.params['adaptive']:
                T, S = Solvers.weight_approx(base_tensor, self.X, self.r)
                tmp = base_tensor - T @ S
                err = torch.linalg.norm(tmp @ self.X) / torch.linalg.norm(tmp)
                mu *= err**2
            
            B, A = Solvers.weight_approx_with_regularization(base_tensor, self.X, self.r, mu)
            
            
            if self.params['log_norms']:
                print('weighted', torch.linalg.norm((base_tensor - B @ A) @ self.X), file=sys.stderr)
                print('non weighted', torch.linalg.norm((base_tensor - B @ A)), file=sys.stderr)
                print('X', torch.linalg.norm(self.X, 2), file=sys.stderr)
                print('mu', mu, file=sys.stderr)
            
        elif self.compress_strategy == "svd_llm":
            B, A = Solvers.svd_llm_method(base_tensor, self.X, self.r)
            
        elif self.compress_strategy == "svd_llm_2":
            B, A = Solvers.svd_llm_2_method(base_tensor, self.X, self.r)
            
        elif self.compress_strategy == "asvd":
            B, A = Solvers.asvd_method(base_tensor, self.X, self.r, self.params['alpha'])
            
        else:
            raise ValueError("Compression strategy is not defined.")
        
        if self.X != None:
            del self.X
        self.adapter_A.copy_(A.T)
        self.adapter_B.copy_(B.T)
        
    
    def forward(self, X):
        if X.dtype != self.adapter_A.data.dtype:
            X = X.to(self.adapter_A.data.dtype)
        return (X @ self.adapter_A) @ self.adapter_B
    
    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}(\n"
            f"  adapter_A=Parameter(shape={self.adapter_A.shape}, dtype={self.adapter_A.dtype}, requires_grad={self.adapter_A.requires_grad}),\n"
            f"  adapter_B=Parameter(shape={self.adapter_B.shape}, dtype={self.adapter_B.dtype}, requires_grad={self.adapter_B.requires_grad}),\n"
            f"  r={self.r},\n"
            f"  compress_strategy='{self.compress_strategy}'\n"
            f"  params={self.params}\n"
            f")"
        )
        return repr_str
