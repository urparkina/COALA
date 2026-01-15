import pytest
import torch
from lib.Solvers import asvd_method
import numpy as np
from numpy.testing import assert_allclose


def test_asvd_method_basic():
    W = torch.randn(20, 10)
    X = torch.randn(10, 100)
    r = 5
    alpha = 0.5

    U, SV = asvd_method(W, X, r, alpha)

    assert U.shape == (20, r)
    assert SV.shape == (r, 10)


def test_asvd_method_reconstruction():
    torch.manual_seed(42)
    W = torch.randn(8, 8)
    X = torch.randn(8, 100)
    r = 4
    alpha = 1.0

    U, SV = asvd_method(W, X, r, alpha)
    W_approx = U @ SV

    error = torch.norm(W - W_approx) / torch.norm(W)
    assert error < 0.5
    
    
def test_asvd_method_alpha_effect():
    W = torch.eye(5)
    X = torch.randn(5, 100)
    X[:, 0] *= 10
    r = 2

    U1, SV1 = asvd_method(W, X, r, alpha=1.0)
    U0, SV0 = asvd_method(W, X, r, alpha=0.0)

    assert not torch.allclose(U1, U0, atol=1e-6)
    assert not torch.allclose(SV1, SV0, atol=1e-6)


def test_asvd_method_edge_cases():
    W = torch.randn(8, 6)
    X = torch.randn(6, 100)
    U, SV = asvd_method(W, X, r=6, alpha=0.5)
    assert torch.allclose(U @ SV, W, atol=1e-6)
    

@pytest.mark.parametrize("m,n", [(10, 10), (10, 5), (5, 10)])
def test_asvd_method_different_shapes(m, n):
    W = torch.randn(m, n)
    X = torch.randn(n, 100)
    r = min(m, n) // 2

    U, SV = asvd_method(W, X, r, alpha=0.5)
    assert U.shape == (m, r)
    assert SV.shape == (r, n)
    
    
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_asvd_method_gpu():
    torch.manual_seed(42)
    W = torch.randn(10, 8)
    X = torch.randn(8, 100)
    r = 5
    alpha = 0.5
    
    W_gpu = W.cuda()
    X_gpu = X.cuda()
    
    U_gpu, SV_gpu = asvd_method(W_gpu, X_gpu, r, alpha)
    W_approx_gpu = U_gpu @ SV_gpu
    
    assert W_approx_gpu.shape == W.shape
    assert torch.linalg.matrix_rank(W_approx_gpu.cpu()) <= r
    assert W_approx_gpu.device.type == 'cuda'
    assert U_gpu.device.type == 'cuda'
    assert SV_gpu.device.type == 'cuda'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_asvd_method_cpu_gpu_equivalence():
    torch.manual_seed(42)
    W = torch.randn(6, 6)
    X = torch.randn(6, 50)
    r = 3
    alpha = 0.7
    
    U_cpu, SV_cpu = asvd_method(W, X, r, alpha)
    W_approx_cpu = U_cpu @ SV_cpu
    
    U_gpu, SV_gpu = asvd_method(W.cuda(), X.cuda(), r, alpha)
    W_approx_gpu = (U_gpu @ SV_gpu).cpu()
    
    assert torch.allclose(W_approx_cpu, W_approx_gpu, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_asvd_method_large_matrix_gpu():
    torch.manual_seed(42)
    W = torch.randn(256, 128)
    X = torch.randn(128, 1000)
    r = 32
    alpha = 0.3
    
    W_gpu = W.cuda()
    X_gpu = X.cuda()
    
    U_gpu, SV_gpu = asvd_method(W_gpu, X_gpu, r, alpha)
    W_approx_gpu = U_gpu @ SV_gpu
    
    assert W_approx_gpu.shape == W.shape
    assert W_approx_gpu.device.type == 'cuda'