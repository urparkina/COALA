import itertools
import math

import pytest
import torch


from lib.TSQR import qr_one_device



@pytest.fixture(params=['cpu'] + (['cuda'] if torch.cuda.is_available() else []),
                scope='module')
def device(request):
    return torch.device(request.param)


def _is_upper_triangular(mat: torch.Tensor, *, atol=1e-6) -> bool:
    if mat.ndim != 2:
        return False
    tri_mask = torch.tril(torch.ones_like(mat), diagonal=-1).bool()
    return torch.all(torch.abs(mat[tri_mask]) < atol)


def _check_r_factor(X_full: torch.Tensor, R: torch.Tensor, *, atol=1e-4, rtol=1e-4):
    m = X_full.shape[0]
    assert R.shape == (m, m), "Неверная форма R"
    assert _is_upper_triangular(R), "R не верхнетреугольная"

    gram_expected = X_full @ X_full.T    
    gram_got      = R.T @ R      
    
    print(gram_got.cpu() - gram_expected.cpu())
    assert torch.allclose(gram_got.cpu(),
                        gram_expected.cpu(),
                        atol=atol,
                        rtol=rtol), "R не удовлетворяет равенству RᵀR = X Xᵀ"


def test_single_matrix_matches_torch(device):
    torch.manual_seed(0)
    m, n = 6, 4
    X = torch.randn(m, n)

    _, r_expected = torch.linalg.qr(X.T, mode='r')
    r_got = qr_one_device(device, [X])

    assert r_got.shape == r_expected.shape

    assert _is_upper_triangular(r_got)

    assert torch.allclose(r_got.cpu(), r_expected, atol=1e-6, rtol=1e-6)
    

def test_multiple_blocks_matches_torch(device):
    torch.manual_seed(42)

    m = 8
    widths = [10, 15, 12, 17]
    blocks = [torch.randn(m, w) for w in widths]

    X_full = torch.cat(blocks, dim=1)
    _, r_expected = torch.linalg.qr(X_full.T, mode='r')

    r_got = qr_one_device(device, iter(blocks))

    assert r_got.shape == r_expected.shape, "Неверная форма результата"
    assert _is_upper_triangular(r_got),      "Матрица R не верх-треугольная"
    _check_r_factor(X_full, r_got)


def test_generator_input(device):
    m, n = 5, 3
    X = torch.randn(m, n)

    mats_iter = (mat for mat in [X])
    out = qr_one_device(device, mats_iter)

    assert _is_upper_triangular(out)
    assert out.shape == torch.linalg.qr(X.T, mode='r')[1].shape


def test_empty_input_raises(device):
    with pytest.raises(ValueError, match="Iterator is empty"):
        qr_one_device(device, [])


def test_non_2d_input_raises(device):
    bad_tensor = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match="Only matrices"):
        qr_one_device(device, [bad_tensor])


def test_mismatched_rows_raises(device):
    X1 = torch.randn(4, 2)
    X2 = torch.randn(5, 3)
    with pytest.raises(ValueError, match="same number of rows"):
        qr_one_device(device, [X1, X2])