import numpy as np

def numdiff(x, f, eps=0.00001):
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


