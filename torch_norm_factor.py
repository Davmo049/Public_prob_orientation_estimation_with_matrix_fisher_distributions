import torch


def _horner(arr, x):
    z = torch.empty(x.shape, dtype=x.dtype, device=x.device).fill_(arr[0])
    for i in range(1, len(arr)):
        z.mul_(x).add_(arr[i])
    return z

torch_bessel0_a  = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2][::-1]
torch_bessel0_b  = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2, -0.2057706e-1, 0.2635537e-1, -0.1647633e-1, 0.392377e-2][::-1]
def bessel0(x): # always supressed by exp(x)
    # x is of size (-1)
    abs_x = torch.abs(x)
    mask = abs_x <= 3.75
    e1 = _horner(torch_bessel0_a, (abs_x/3.75)**2) / torch.exp(abs_x)
    e2 = _horner(torch_bessel0_b, 3.75/abs_x)/torch.sqrt(abs_x)
    e2[mask] = e1[mask]
    return e2

def torch_integral(f, v, from_x, to_x, N):
    with torch.no_grad():
        # computes ret_i = \int_{from_x}^{to_x} f(x,v_i)
        # where N is number of trapezoids + 1 per v_i
        rangee = torch.arange(N,dtype=v.dtype, device=v.device)
        x = (rangee*((to_x-from_x)/(N-1))+from_x).view(1, N)
        weights = torch.empty((1, N), dtype=v.dtype, device=v.device).fill_(1)
        weights[0, 0] = 1/2
        weights[0, -1] = 1/2
        y = f(x, v)
        return torch.sum(y*weights, dim=1)*(to_x-from_x)/(N-1)

def integrand_CF(x, s):
    # x is (1, N)
    # s is (-1, 3)
    # return (-1, N)
    # s is sorted from large to small
    f1 = (s[:, 1]-s[:, 2])/2
    f2 = (s[:, 1]+s[:, 2])/2
    a1 = f1.view(-1,1)*(1-x).view(1, -1)
    a2 = f2.view(-1,1)*(1+x).view(1, -1)
    a3 = (s[:,2]+s[:,0]).view(-1,1)*(x-1).view(1, -1)
    i1 = bessel0(a1)
    i2 = bessel0(a2)
    i3 = torch.exp(a3)
    ret = i1*i2*i3
    return ret


def integrand_Cdiff(x, s):
    s2 = s[:, 0]
    s1 = torch.max(s[:, 1:], dim=1).values
    s0 = torch.min(s[:, 1:], dim=1).values
    f1 = (s1-s0)/2
    f2 = (s1+s0)/2
    a1 = f1.view(-1,1)*(1-x).view(1, -1)
    a2 = f2.view(-1,1)*(1+x).view(1, -1)
    a3 = (s0+s2).view(-1,1)*(x-1).view(1, -1)
    i1 = bessel0(a1)
    i2 = bessel0(a2)
    i3 = x.view(1, -1)
    i4 = torch.exp(a3)
    return i1*i2*i3*i4


class class_logC_F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        N = 512
        # input is (..., 3) correspond to SINGULAR VALUES of F (NOT Lambda)
        shape = input.shape
        input_v = input.view(-1, 3)
        factor = 1/2*torch_integral(integrand_CF, input_v,-1, 1, N)
        log_factor = torch.log(factor)
        log_supress = torch.sum(input_v, dim=1)
        ctx.save_for_backward(input, factor)
        return (log_factor+log_supress).view(shape[:-1])

    @staticmethod
    def backward(ctx, grad):
        S, factor = ctx.saved_tensors
        S = S.view(-1, 3)
        N = 512
        ret = torch.empty((S.shape[0], 3), dtype=S.dtype, device=S.device)
        for i in range(3):
            cv = torch.cat((S[:, i:], S[:, :i]), dim=1)
            ret[:, i] = 1/2*torch_integral(integrand_Cdiff, cv,-1, 1, N)
        ret /= factor.view(-1, 1)
        ret *= grad.view(-1, 1)
        return ret.view((*grad.shape, 3))

logC_F = class_logC_F.apply

def main():
    import numpy as np
    import matplotlib.pyplot as plt
    ggg = np.arange(0, 100, 1)
    outs = []
    for xx in ggg:
        v = np.array([1.0, 0.5, -0.2])*xx
        vt = torch.tensor(v, requires_grad=True).view(1, 3)
        o = logC_F(vt)
        l = torch.sum(o.flatten())
        l.backward()
        outs.append(o.detach().numpy()[0])
    plt.plot(ggg, outs)
    plt.show()

if __name__ == '__main__':
    main()
