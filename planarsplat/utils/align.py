# modified from https://github.com/microsoft/MoGe/blob/main/moge/utils/alignment.py
from typing import *
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.types

def scatter_min(size: int, dim: int, index: torch.LongTensor, src: torch.Tensor) -> torch.return_types.min:
    "Scatter the minimum value along the given dimension of `input` into `src` at the indices specified in `index`."
    shape = src.shape[:dim] + (size,) + src.shape[dim + 1:]
    minimum = torch.full(shape, float('inf'), dtype=src.dtype, device=src.device).scatter_reduce(dim=dim, index=index, src=src, reduce='amin', include_self=False)
    minimum_where = torch.where(src == torch.gather(minimum, dim=dim, index=index))
    indices = torch.full(shape, -1, dtype=torch.long, device=src.device)
    indices[(*minimum_where[:dim], index[minimum_where], *minimum_where[dim + 1:])] = minimum_where[dim]
    return torch.return_types.min((minimum, indices))
    

def split_batch_fwd(fn: Callable, chunk_size: int, *args, **kwargs):
    batch_size = next(x for x in (*args, *kwargs.values()) if isinstance(x, torch.Tensor)).shape[0]
    n_chunks = batch_size // chunk_size + (batch_size % chunk_size > 0)
    splited_args = tuple(arg.split(chunk_size, dim=0) if isinstance(arg, torch.Tensor) else [arg] * n_chunks for arg in args)
    splited_kwargs = {k: [v.split(chunk_size, dim=0) if isinstance(v, torch.Tensor) else [v] * n_chunks] for k, v in kwargs.items()}
    results = []
    for i in range(n_chunks):
        chunk_args = tuple(arg[i] for arg in splited_args)
        chunk_kwargs = {k: v[i] for k, v in splited_kwargs.items()}
        results.append(fn(*chunk_args, **chunk_kwargs))

    if isinstance(results[0], tuple):
        return tuple(torch.cat(r, dim=0) for r in zip(*results))
    else:
        return torch.cat(results, dim=0)

def _pad_inf(x_: torch.Tensor):
    return torch.cat([torch.full_like(x_[..., :1], -torch.inf), x_, torch.full_like(x_[..., :1], torch.inf)], dim=-1)


def _pad_cumsum(cumsum: torch.Tensor):
    return torch.cat([torch.zeros_like(cumsum[..., :1]), cumsum, cumsum[..., -1:]], dim=-1)


def _compute_residual(a: torch.Tensor, xyw: torch.Tensor, trunc: float):
    return a.mul(xyw[..., 0]).sub_(xyw[..., 1]).abs_().mul_(xyw[..., 2]).clamp_max_(trunc).sum(dim=-1)



def align(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, trunc: Optional[Union[float, torch.Tensor]] = None, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
    """
    If trunc is None, solve `min sum_i w_i * |a * x_i - y_i|`, otherwise solve `min sum_i min(trunc, w_i * |a * x_i - y_i|)`.
    
    w_i must be >= 0.

    ### Parameters:
    - `x`: tensor of shape (..., n)
    - `y`: tensor of shape (..., n)
    - `w`: tensor of shape (..., n)
    - `trunc`: optional, float or tensor of shape (..., n) or None

    ### Returns:
    - `a`: tensor of shape (...), differentiable
    - `loss`: tensor of shape (...), value of loss function at `a`, detached
    - `index`: tensor of shape (...), where a = y[idx] / x[idx]
    """
    if trunc is None:
        x, y, w = torch.broadcast_tensors(x, y, w)
        sign = torch.sign(x)
        x, y = x * sign, y * sign
        y_div_x = y / x.clamp_min(eps)
        y_div_x, argsort = y_div_x.sort(dim=-1)

        wx = torch.gather(x * w, dim=-1, index=argsort)
        derivatives = 2 * wx.cumsum(dim=-1) - wx.sum(dim=-1, keepdim=True)
        search = torch.searchsorted(derivatives, torch.zeros_like(derivatives[..., :1]), side='left').clamp_max(derivatives.shape[-1] - 1)

        a = y_div_x.gather(dim=-1, index=search).squeeze(-1)
        index = argsort.gather(dim=-1, index=search).squeeze(-1)
        loss = (w * (a[..., None] * x - y).abs()).sum(dim=-1)
        
    else:
        # Reshape to (batch_size, n) for simplicity
        x, y, w = torch.broadcast_tensors(x, y, w)
        batch_shape = x.shape[:-1]
        batch_size = math.prod(batch_shape)
        x, y, w = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]), w.reshape(-1, w.shape[-1])

        sign = torch.sign(x)
        x, y = x * sign, y * sign
        wx, wy = w * x, w * y
        xyw = torch.stack([x, y, w], dim=-1)    # Stacked for convenient gathering

        y_div_x = A = y / x.clamp_min(eps)
        B = (wy - trunc) / wx.clamp_min(eps)
        C = (wy + trunc) / wx.clamp_min(eps)
        with torch.no_grad():
            # Caculate prefix sum by orders of A, B, C    
            A, A_argsort = A.sort(dim=-1)
            Q_A = torch.cumsum(torch.gather(wx, dim=-1, index=A_argsort), dim=-1)
            A, Q_A = _pad_inf(A), _pad_cumsum(Q_A)    # Pad [-inf, A1, ..., An, inf] and [0, Q1, ..., Qn, Qn] to handle edge cases.

            B, B_argsort = B.sort(dim=-1)
            Q_B = torch.cumsum(torch.gather(wx, dim=-1, index=B_argsort), dim=-1)
            B, Q_B = _pad_inf(B), _pad_cumsum(Q_B)

            C, C_argsort = C.sort(dim=-1)
            Q_C = torch.cumsum(torch.gather(wx, dim=-1, index=C_argsort), dim=-1)
            C, Q_C = _pad_inf(C), _pad_cumsum(Q_C)
            
            # Caculate left and right derivative of A
            j_A = torch.searchsorted(A, y_div_x, side='left').sub_(1)
            j_B = torch.searchsorted(B, y_div_x, side='left').sub_(1)
            j_C = torch.searchsorted(C, y_div_x, side='left').sub_(1)
            left_derivative = 2 * torch.gather(Q_A, dim=-1, index=j_A) - torch.gather(Q_B, dim=-1, index=j_B) - torch.gather(Q_C, dim=-1, index=j_C)
            j_A = torch.searchsorted(A, y_div_x, side='right').sub_(1)
            j_B = torch.searchsorted(B, y_div_x, side='right').sub_(1)
            j_C = torch.searchsorted(C, y_div_x, side='right').sub_(1)
            right_derivative = 2 * torch.gather(Q_A, dim=-1, index=j_A) - torch.gather(Q_B, dim=-1, index=j_B) - torch.gather(Q_C, dim=-1, index=j_C)

            # Find extrema
            is_extrema = (left_derivative < 0) & (right_derivative >= 0)
            is_extrema[..., 0] |= ~is_extrema.any(dim=-1)                       # In case all derivatives are zero, take the first one as extrema.
            where_extrema_batch, where_extrema_index = torch.where(is_extrema)          

            # Calculate objective value at extrema
            extrema_a = y_div_x[where_extrema_batch, where_extrema_index]               # (num_extrema,)
            MAX_ELEMENTS = 4096 ** 2      # Split into small batches to avoid OOM in case there are too many extrema.(~1G)
            SPLIT_SIZE = MAX_ELEMENTS // x.shape[-1]
            extrema_value = torch.cat([
                _compute_residual(extrema_a_split[:, None], xyw[extrema_i_split, :, :], trunc)
                for extrema_a_split, extrema_i_split in zip(extrema_a.split(SPLIT_SIZE), where_extrema_batch.split(SPLIT_SIZE))
            ])          # (num_extrema,)
            
            # Find minima among corresponding extrema
            minima, indices = scatter_min(size=batch_size, dim=0, index=where_extrema_batch, src=extrema_value)        # (batch_size,)
            index = where_extrema_index[indices]

        a = torch.gather(y, dim=-1, index=index[..., None]) / torch.gather(x, dim=-1, index=index[..., None]).clamp_min(eps)
        a = a.reshape(batch_shape)
        loss = minima.reshape(batch_shape)
        index = index.reshape(batch_shape)

    return a, loss, index


def align_depth_scale(depth_src: torch.Tensor, depth_tgt: torch.Tensor, weight: Optional[torch.Tensor], trunc: Optional[Union[float, torch.Tensor]] = None):
    """
    Align `depth_src` to `depth_tgt` with given constant weights. 

    ### Parameters:
    - `depth_src: torch.Tensor` of shape (..., N)
    - `depth_tgt: torch.Tensor` of shape (..., N)

    """
    scale, _, _ = align(depth_src, depth_tgt, weight, trunc)

    return scale