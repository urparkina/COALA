import pytest
import torch
from lib.Solvers import weight_approx
import numpy as np
from numpy.testing import assert_allclose



def test_weight_approx_basic():
    torch.manual_seed(42)
    d, m, n = 10, 5, 20
    W = torch.randn(m, d)
    X = torch.randn(d, n)
    r = 3
    
    A, B = weight_approx(W, X, r)
    
    assert A.shape == (m, r)
    assert B.shape == (r, d)


def test_weight_approx_reconstruction():
    torch.manual_seed(42)
    d, m, n = 10, 5, 20
    W = torch.randn(m, d)
    X = torch.randn(d, n)
    r = min(W.shape)
    
    A, B = weight_approx(W, X, r)
    W_approx = A @ B
    
    error = torch.norm((W - W_approx) @ X, 'fro')
    original_norm = torch.norm(W @ X, 'fro')
    assert error < 1e-5 * original_norm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_weight_approx_gpu():
    torch.manual_seed(42)
    d, m, n = 10, 5, 20
    W = torch.randn(m, d)
    X = torch.randn(d, n)
    r = 3
    
    W_gpu = W.cuda()
    X_gpu = X.cuda()
    
    A_gpu, B_gpu = weight_approx(W_gpu, X_gpu, r)
    W_approx_gpu = A_gpu @ B_gpu
    
    assert W_approx_gpu.shape == W.shape
    assert torch.linalg.matrix_rank(W_approx_gpu.cpu()) <= r
    assert W_approx_gpu.device.type == 'cuda'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_weight_approx_cpu_gpu_equivalence():
    torch.manual_seed(42)
    d, m, n = 10, 5, 20
    W = torch.randn(m, d)
    X = torch.randn(d, n)
    r = 4
    
    A_cpu, B_cpu = weight_approx(W, X, r)
    A_gpu, B_gpu = weight_approx(W.cuda(), X.cuda(), r)
    W_approx_cpu = A_cpu @ B_cpu
    W_approx_gpu = (A_gpu @ B_gpu).cpu()
    
    assert_allclose(W_approx_cpu.numpy(), W_approx_gpu.numpy(), rtol=1e-5, atol=1e-5)


def test_weight_approx_rank_zero():
    torch.manual_seed(42)
    d, m, n = 10, 5, 20
    W = torch.randn(m, d)
    X = torch.randn(d, n)
    r = 0
    
    A, B = weight_approx(W, X, r)
    W_approx = A @ B
    
    assert W_approx.shape == W.shape
    assert torch.allclose(W_approx, torch.zeros_like(W))
    
    
def test_weight_approx_minimize_norm_effect():
    torch.manual_seed(42)
    d, m, n = 30, 20, 5
    W = torch.randn(m, d)
    X = torch.randn(d, n)
    r = 3
    
    A, B = weight_approx(W, X, r)
    C, D = weight_approx(W, X, r, minimize_norm=True)
    W_approx_default = A @ B
    W_approx_min_norm = C @ D
    
    error_default = torch.norm((W - W_approx_default) @ X, 'fro')
    error_min_norm = torch.norm((W - W_approx_min_norm) @ X, 'fro')
    
    norm_default = torch.norm(W_approx_default, 'fro')
    norm_min_norm = torch.norm(W_approx_min_norm, 'fro')
    
    assert norm_min_norm < norm_default, "minimize_norm should reduce the Frobenius norm"
    assert torch.abs(error_min_norm - error_default) < 0.001, (
        "Error with minimize_norm shouldn't be much worse than without"
    )
    
    assert norm_min_norm <= norm_default, (
        "minimize_norm doesn't work"
    )
    
    assert torch.linalg.matrix_rank(W_approx_default) <= r
    assert torch.linalg.matrix_rank(W_approx_min_norm) <= r
    

def test_qr_factoring():
    torch.manual_seed(42)
    W = torch.eye(5)
    X = torch.triu(torch.randn(5, 5), diagonal=0)
    r = 3

    A, B = weight_approx(W, X, r)
    W_approx = A @ B
    error = torch.norm((W - W_approx) @ X, 'fro')
    print(error)
    assert error <= 0.5