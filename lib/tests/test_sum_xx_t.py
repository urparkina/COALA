import math
import sys
import gc
import pytest
import torch
from lib.TSQR import sum_x_xt


def _make_mats(m=4, n=6, k=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(m, n, generator=g) for _ in range(k)]


def test_sum_x_xt_cpu():
    mats = _make_mats()
    expected = sum(X @ X.T for X in mats)
    out = sum_x_xt("cpu", iter(mats))
    assert torch.allclose(out, expected, atol=1e-6)


def test_empty_iterator_raises():
    with pytest.raises(ValueError, match="empty"):
        sum_x_xt("cpu", [])
        
        
def test_dim_and_shape_validation():
    with pytest.raises(ValueError, match="Only matrices"):
        sum_x_xt("cpu", [torch.randn(3)])

    m1, m2 = torch.randn(3, 4), torch.randn(5, 4)
    with pytest.raises(ValueError, match="same number of rows"):
        sum_x_xt("cpu", [m1, m2])


def test_original_tensors_stay_on_cpu_and_unchanged():
    mats = _make_mats()
    clones = [x.clone() for x in mats]

    _ = sum_x_xt("cpu", iter(mats))

    for orig, ref in zip(mats, clones):
        assert orig.device.type == "cpu"
        assert torch.allclose(orig, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_memory_is_small_and_released():
    device = torch.device("cuda")

    m, n, k = 1024, 1024, 5000
    seed = 0

    def mats_generator():
        g = torch.Generator().manual_seed(seed)
        for _ in range(k):
            yield torch.randn(m, n, generator=g)

    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated(device)

    result = sum_x_xt(device, mats_generator())

    gc.collect()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated(device)

    expected_bytes = result.nelement() * result.element_size()
    overhead = after - expected_bytes

    assert result.shape == (m, m)
    assert result.device.type == "cuda"
