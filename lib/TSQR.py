import torch
from typing import Iterable

def sum_x_xt(device, mats) -> torch.Tensor:
    """
    Compute Sigma X X^T for a stream of 2-D tensors.

    device : target device ('cuda:0', 'cpu', ...)
    mats   : iterable of matrices with the same row count m

    Returns: m x m Gram matrix
    Raises : ValueError on empty input, non-2-D tensors, or mismatched rows
    """
    mats = iter(mats)

    try:
        first = next(mats)
    except StopIteration:
        raise ValueError("Iterator is empty")

    if first.ndim != 2:
        raise ValueError("Only matrices")

    first = first.to(device)
    result = first @ first.T
    del first

    for X in mats:
        if X.shape[0] != result.shape[0]:
            raise ValueError("All matrices must have the same number of rows (m)")
        X = X.to(device)
        result = result + X @ X.T
        del X

    return result


def qr_one_device(device, mats) -> torch.Tensor:
    """
    Incremental, communication-optimal QR (R factor only) of X^T.

    Implements the single-device version of the algorithm from  
    J. Demmel, L. Grigori, M. Hoemmen, J. Langou,  
    “Communication-Optimal Parallel and Sequential QR and LU Factorizations,”  
    SIAM J. Sci. Comput., 34(1):A206-A239, 2012.

    device : computation device
    mats   : 2-D tensors, same row count m

    Returns: upper-triangular R (m x m)
    Raises : ValueError on empty input, non-2-D tensors, or mismatched rows
    """
    mats = iter(mats)

    try:
        first = next(mats)
    except StopIteration:
        raise ValueError("Iterator is empty")

    if first.ndim != 2:
        raise ValueError("Only matrices")

    first = first.to(device)
    _, result = torch.linalg.qr(first.T, mode='r')
    del first

    for X in mats:
        if X.shape[0] != result.shape[0]:
            raise ValueError("All matrices must have the same number of rows (m)")
        X = X.to(device)
        _, result = torch.linalg.qr(torch.cat((result.T, X), dim=1).T, mode='r')
        del X

    return result