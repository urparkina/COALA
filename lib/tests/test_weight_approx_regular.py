import pytest
import torch
from numpy.testing import assert_allclose
from lib.Solvers import weight_approx_with_regularization

@pytest.fixture
def setup_data():
    m, d, n, r = 5, 3, 10, 2
    W = torch.randn(m, d)
    X = torch.randn(d, n)
    mu = 0.1
    return W, X, r, mu

def test_output_shapes(setup_data):
    W, X, r, mu = setup_data
    A, B = weight_approx_with_regularization(W, X, r, mu)
    
    assert A.shape == (W.shape[0], r)
    assert B.shape == (r, X.shape[0])

def test_regularization_effect(setup_data):
    W, X, r, mu = setup_data
    
    for test_mu in [0.0, 0.1, 1.0, 10.0]:
        A, B = weight_approx_with_regularization(W, X, r, test_mu)
        W_approx = A @ B
        
        assert torch.isfinite(W_approx).all()
        assert W_approx.shape == W.shape

def test_zero_regularization(setup_data):
    W, X, r, mu = setup_data
    A_zero, B_zero = weight_approx_with_regularization(W, X, r, 0.0)
    A_reg, B_reg = weight_approx_with_regularization(W, X, r, 1e-10)
    
    assert torch.allclose(A_zero, A_reg, atol=1e-5)
    assert torch.allclose(B_zero, B_reg, atol=1e-5)
    

def test_device_preservation(setup_data):
    W, X, r, mu = setup_data
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    for device in devices:
        W_dev = W.to(device)
        X_dev = X.to(device)
        A, B = weight_approx_with_regularization(W_dev, X_dev, r, mu)
        
        assert A.device == W_dev.device
        assert B.device == W_dev.device


def test_dtype_preservation(setup_data):
    W, X, r, mu = setup_data
    for dtype in [torch.float32, torch.float64]:
        W_dt = W.to(dtype)
        X_dt = X.to(dtype)
        A, B = weight_approx_with_regularization(W_dt, X_dt, r, mu)
        
        assert A.dtype == dtype
        assert B.dtype == dtype