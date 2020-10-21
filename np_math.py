import numpy as np

def numdiff(x, f, eps=10e-5):
    # computes a numerical diff of f at x
    xflat = x.flatten()
    diff = []
    for i in range(xflat.shape[0]):
        p = np.zeros(xflat.shape)
        p[i] = eps
        xpp = np.copy(xflat) + p
        xpm = np.copy(xflat) - p
        fp = f(xpp.reshape(x.shape))
        fm = f(xpm.reshape(x.shape))
        diff.append((fp-fm)/(2*eps))
    return np.array(diff).reshape(x.shape)

def _horner(arr, pos):
    z = 0
    for val in arr:
        z = z*pos +val
    return z

b0_a = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2][::-1]
b0_b = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2, -0.2057706e-1, 0.2635537e-1, -0.1647633e-1, 0.392377e-2][::-1]

def bessel0(x):
    abs_x = np.abs(x)
    if abs_x <= 3.75:
        return _horner(b0_a, abs_x*abs_x/(3.75*3.75))
    else:
        return (np.exp(abs_x)/np.sqrt(abs_x))*_horner(b0_b, 3.75/abs_x)

b1_a = [0.5, 0.87890594, 0.51498869, 0.15084934, 0.2658733e-1, 0.301532e-2, 0.32411e-3][::-1]
b1_b = [0.39894228, -0.3988024e-1, -0.362018e-2, 0.163801e-2, -0.1031555e-1, 0.2282967e-1, -0.2895312e-1, 0.1787654e-1, -0.420059e-2][::-1]

def bessel1(x):
    abs_x = np.abs(x)
    if abs_x <= 3.75:
        return x*_horner(b1_a, abs_x*abs_x/(3.75*3.75))
    else:
        sign_x = np.sign(x)
        return sign_x*(np.exp(abs_x)/np.sqrt(abs_x))*_horner(b1_b, 3.75/abs_x)

def batch_bessel0(x):
    abs_x = np.abs(x)
    mask = abs_x <= 3.75
    e1 = _horner(b0_a, (abs_x/3.75)**2)
    e2 = (np.exp(abs_x)/np.sqrt(abs_x))*_horner(b0_b, 3.75/abs_x)
    e2[mask] = e1[mask]
    return e2

def batch_bessel0_supress(x):
    abs_x = np.abs(x)
    mask = abs_x <= 3.75
    e1 = _horner(b0_a, (abs_x/3.75)**2)/np.exp(abs_x)
    e2 = 1/np.sqrt(abs_x)*_horner(b0_b, 3.75/abs_x)
    e2[mask] = e1[mask]
    return e2


def batch_bessel1(x):
    abs_x = np.abs(x)
    mask = abs_x <= 3.75
    e1 = x*_horner(b1_a, abs_x*abs_x/(3.75*3.75))
    sign_x = np.sign(x)
    e2 = sign_x*(np.exp(abs_x)/np.sqrt(abs_x))*_horner(b1_b, 3.75/abs_x)
    e2[mask] = e1[mask]
    return e2

def numintegral(f, from_x, to_x, N, dtype):
    # trapezoid
    x = np.arange(N, dtype=dtype)*(to_x-from_x)/(N-1)+from_x
    weights = np.ones(x.shape, dtype=dtype)
    weights[0]= 1/2
    weights[-1] = 1/2
    y = f(x)
    return (to_x-from_x)*np.sum(y*weights)/(N-1)


def batch_forward(x, s):
    fact1 = (s[2]-s[1])/2
    fact2 = (s[2]+s[1])/2
    a1 = fact1*(1-x)
    a2 = fact2*(1-x)
    f1 = batch_bessel0(fact1*(1-x))
    f2 = batch_bessel0(fact2*(1+x))
    f3 = np.exp(s[0]*x)
    f3 = np.exp(s[0]*x)

    return f1*f2*f3


def forward(s, N=1000):
    sp = sorted(s)[::-1] # largest first
    f = lambda x: batch_forward(x, sp)
    return 1/2*numintegral(f, -1, 1, N, s.dtype)

def batch_backward(x, s):
    fact1 = (s[2]-s[1])/2
    fact2 = (s[2]+s[1])/2
    f1 = batch_bessel0(fact1*(1-x))
    f2 = batch_bessel0(fact2*(1+x))
    f3 = np.exp(s[0]*x)
    v = x*f1*f2*f3
    return v

def backward_single(s, N, dtype):
    f = lambda x: batch_backward(x, s)
    return 1/2*numintegral(f, -1, 1, N, dtype=dtype)


def backward(s, N=1000):
    ret = np.empty((3), dtype=s.dtype)
    for i in range(3):
        sp = list(map(lambda x: s[(x+i)%3], range(3)))
        ret[i] = backward_single(sp, N, s.dtype)
    return ret

def batch_forward_supress(x, s):
    fact1 = (s[1]-s[2])/2
    fact2 = (s[1]+s[2])/2
    a1 = np.abs(fact1*(1-x))
    a2 = np.abs(fact2*(1+x))
    a3 = (s[0]+s[2])*(x-1)
    f1 = batch_bessel0_supress(a1)
    f2 = batch_bessel0_supress(a2)
    f3 = np.exp(a3)
    ret = f1*f2*f3
    return ret

def forward_supress(s, N=1000):
    sp = sorted(s)[::-1] # largest first
    f = lambda x: batch_forward_supress(x, sp)
    return np.log(1/2*numintegral(f, -1, 1, N, s.dtype))+np.sum(s)

def batch_backward_supress(x, s):
    s1 = max(s[1], s[2])
    s2 = min(s[1], s[2])
    fact1 = (s1-s2)/2
    fact2 = (s1+s2)/2
    a1 = fact1*(1-x)
    a2 = fact2*(1+x)

    f1 = batch_bessel0_supress(a1)
    f2 = batch_bessel0_supress(a2)
    f3 = np.exp((s[0]+s2)*(x-1))
    return x*f1*f2*f3

def backward_single_supress(s, N, dtype):
    f = lambda x: batch_backward_supress(x, s)
    return 1/2*numintegral(f, -1, 1, N, dtype=dtype)

def backward_supress(s, N=1000):
    factor = np.exp(forward_supress(s, N) - np.sum(s))
    ret = np.empty((3), dtype=s.dtype)
    for i in range(3):
        sp = list(map(lambda x: s[(x+i)%3], range(3)))
        ret[i] = backward_single_supress(sp, N, s.dtype)/factor
    return ret

def log_integral(point, N):
    a = point[1]
    b = point[2]
    c = point[0]
    sm = (c+b)
    fro = np.exp(0)
    to = np.exp(sm)
    xx = np.arange(0, N, dtype=point.dtype)*(to-fro)/(N-1)+fro
    x = np.log(xx)/sm
    weight= np.ones(x.shape, dtype=point.dtype)
    weight[0] = 1/2
    weight[-1] = 1/2
    y = batch_bessel0_supress((a-b)*(1-x)/2)*batch_bessel0_supress((a+b)*(1+x)/2)/sm
    sv = ((to-fro)/np.exp(b+c))
    intv1 = 1/2*np.sum(weight*y)*sv/(N-1)
    return intv1


if __name__ == '__main__':
    main()
