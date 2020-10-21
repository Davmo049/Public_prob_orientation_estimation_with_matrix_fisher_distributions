import torch
import numpy as np

import torch_math

import torch_norm_factor
import np_math


def eval_approximation_accuracy()
    X = np.random.seed(9001)
    T = 10
    bs = 100
    err = []
    err_grad_norm = []
    for _ in range(T):
        point = np.random.normal(size=(bs, 3))
        point /= np.linalg.norm(point, axis=1).reshape(bs, 1)
        r = np.random.uniform(0,50, size=(bs, 1))
        s = np.sign(np.random.uniform(-1,1, size=(bs)))
        point = (point * r)
        point = np.abs(point)
        point = np.sort(point)[:,::-1]
        point[:, 2] *= s
        point_t = torch.tensor(point.copy(), dtype=torch.float32, requires_grad=True)
        c = torch_norm_factor.logC_F(point_t)
        l = torch.sum(c)
        l.backward()
        gr = point_t.grad
        nd = torch_math.numdiff(lambda x: torch_norm_factor.logC_F(x), point_t)
        for bi in range(bs):
            c_t = np_math.forward_supress(point[bi].astype(np.longdouble), 2**14)
            gr_t = np_math.backward_supress(point[bi].astype(np.longdouble), 2**14)
            c_idx = c[bi].detach().numpy()
            gr_idx = gr[bi].detach().numpy()
            err_c = np.abs(c_t - c_idx)
            err_norm_c = np.linalg.norm(gr_t - gr_idx)
            err.append(err_c)
            err_grad_norm.append(err_norm_c)
    import matplotlib.pyplot as plt
    print('max error: {}'.format(np.max(err)))
    print('max error grad norm: {}'.format(np.max(err_grad_norm)))
    plt.hist(err_grad_norm, 100)
    plt.show()
    plt.hist(err, 100)
    plt.show()

if __name__ == '__main__':
    eval_approximation_accuracy()
