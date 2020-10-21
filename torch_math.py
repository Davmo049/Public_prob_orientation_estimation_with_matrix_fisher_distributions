import torch
import numpy as np

def numdiff(f, X, eps=10e-5, batch_dims=1):
    # f is function
    # X is input
    # eps is step to use
    # batch dim is number of first dimensions to ignore as part of a batch (independent)
    X = X.detach()
    bs = np.prod(X.shape[:batch_dims])
    dims_per_batch = np.prod(X.shape[batch_dims:])
    X_like = X.view(bs, dims_per_batch)
    fx = f(X).view(bs)
    diff = torch.empty((bs, dims_per_batch), dtype=X.dtype, device=X.device)
    for i in range(dims_per_batch):
        X_eps = X_like.clone()
        X_eps[:, i] += eps
        diff[:, i] = (f(X_eps.view(*X.shape)).view(bs)-fx)/eps
    return diff.view(*X.shape)
