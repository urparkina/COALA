import pytest
import torch
from lib.Solvers import svd_llm_method, svd_llm_2_method
import numpy as np
from numpy.testing import assert_allclose



def test_svd_llm_basic():
    torch.manual_seed(42)
    d, m, n = 10, 5, 20
    W = torch.randn(m, d)
    X = torch.randn(d, n)
    r = 3
    
    A, B = svd_llm_2_method(W, X, r)
    
    assert A.shape == (m, r)
    assert B.shape == (r, d)


def test_svd_llm_approx_reconstruction():
    torch.manual_seed(42)
    d, m, n = 10, 5, 20
    W = torch.randn(m, d)
    X = torch.randn(d, n)
    r = min(W.shape)
    
    A, B = svd_llm_2_method(W, X, r)
    W_approx = A @ B
    
    error = torch.norm((W - W_approx) @ X, 'fro')
    original_norm = torch.norm(W @ X, 'fro')
    assert error < 1e-5 * original_norm


def test_equivalence_weight_approx_and_svd_llm():
    torch.manual_seed(42)
    d, m, n = 10, 8, 20
    W = torch.randn(m, d)
    X = torch.randn(d, n)
    r = 4
    
    A, B = svd_llm_method(W, X, r)
    W_approx_svd_llm = A @ B
    
    A, B = svd_llm_2_method(W, X, r)
    W_approx_svd_llm_2 = A @ B
    
    error = torch.norm(W_approx_svd_llm - W_approx_svd_llm_2, 'fro')
    assert error < 1e-5 * torch.norm(W, 'fro')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_svd_llm_gpu():
    torch.manual_seed(42)
    d, m, n = 10, 5, 20
    W = torch.randn(m, d)
    X = torch.randn(d, n)
    r = 3
    
    W_gpu = W.cuda()
    X_gpu = X.cuda()
    
    A_gpu, B_gpu = svd_llm_2_method(W_gpu, X_gpu, r)
    W_approx_gpu = A_gpu @ B_gpu
    
    assert W_approx_gpu.shape == W.shape
    assert torch.linalg.matrix_rank(W_approx_gpu.cpu()) <= r
    assert W_approx_gpu.device.type == 'cuda'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_svd_llm_cpu_gpu_equivalence():
    torch.manual_seed(42)
    d, m, n = 10, 5, 20
    W = torch.randn(m, d)
    X = torch.randn(d, n)
    r = 4
    
    A_cpu, B_cpu = svd_llm_2_method(W, X, r)
    A_gpu, B_gpu = svd_llm_2_method(W.cuda(), X.cuda(), r)
    W_approx_cpu = A_cpu @ B_cpu
    W_approx_gpu = (A_gpu @ B_gpu).cpu()
    
    assert_allclose(W_approx_cpu.numpy(), W_approx_gpu.numpy(), rtol=1e-5, atol=1e-5)