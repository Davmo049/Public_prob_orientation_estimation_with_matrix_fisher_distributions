import ctypes
import os
import types
import numpy as np

loaded_library = ctypes.CDLL(os.path.join(os.getcwd(), 'build/library.so'))

x = loaded_library.x
x.restype = ctypes.c_double

c_hg = loaded_library.hg
c_hg.restype = ctypes.c_double

c_loghg = loaded_library.loghg
c_loghg.restype = ctypes.c_double

def x(x):
    y = np.array([3.0, 2.0])
    data = y.data
    pointer = (ctypes.c_double*2).from_buffer(data)
    asd = loaded_library.x(pointer)
    print(type(asd))
    print(asd)
    return asd

def test(x, b=np.array([1.0, 0.5, 0.25, 0.125]), m=10):
    asd = np.log(hg(m, 2, 0.5, 2, x*b))
    return asd


def hg(m, alpha, p, q, x, y=None):
    # n is length of x,y
    # len of p,q is related to p_F_q
    # y is NULL
    # alpha = 2
    # test m=10 related to precision
    assert(isinstance(x, np.ndarray))
    assert(isinstance(y, np.ndarray) or y is None)
    assert(isinstance(p, np.ndarray) or isinstance(p, float) or isinstance(p, int))
    assert(isinstance(q, np.ndarray) or isinstance(q, float) or isinstance(q, int))

    p_in = None
    np_in = 1
    if isinstance(p, np.ndarray):
        if len(p) == 0:
            p_in = 0
            np_in = None
        else:
            p_in = (ctypes.c_double*1).from_buffer(p.data)
            np_in = ctypes.c_int(len(p))
    else:
        p_in = ctypes.byref(ctypes.c_double(p))
    nq_in = 1
    if isinstance(q, np.ndarray):
        if len(q) == 0:
            q_in = 0
            np_in = None
        else:
            q_in = (ctypes.c_double*1).from_buffer(q.data)
            nq_in = ctypes.c_int(len(q))
    else:
        q_in = ctypes.byref(ctypes.c_double(q))
    alpha_in = ctypes.c_double(alpha)
    max_in = ctypes.c_int(m)
    x_in = (ctypes.c_double*len(x)).from_buffer(x.data)
    y_in = None
    if y is not None:
        y_in = (ctypes.c_double*len(y)).from_buffer(y.data)
    n_in = ctypes.c_int(len(x))
    return c_hg(max_in, alpha_in, n_in, x_in, y_in, p_in, q_in,np_in, nq_in)


def loghg(m, alpha, p, q, x, y=None):
    # n is length of x,y
    # len of p,q is related to p_F_q
    # y is NULL
    # alpha = 2
    # test m=10 related to precision

    assert(isinstance(x, np.ndarray))
    assert(isinstance(y, np.ndarray) or y is None)
    assert(isinstance(p, np.ndarray) or isinstance(p, float))
    assert(isinstance(q, np.ndarray) or isinstance(p, float))

    p_in = None
    np_in = 1
    if isinstance(p, np.ndarray):
        p_in = (ctypes.c_double*1).from_buffer(p.data)
        np_in = ctypes.c_int(len(p))
    else:
        p_in = ctypes.byref(ctypes.c_double(p))
    nq_in = 1
    if isinstance(q, np.ndarray):
        q_in = (ctypes.c_double*1).from_buffer(q.data)
        nq_in = ctypes.c_int(len(q))
    else:
        q_in = ctypes.byref(ctypes.c_double(q))
    alpha_in = ctypes.c_double(alpha)
    max_in = ctypes.c_int(m)
    x_in = (ctypes.c_double*len(x)).from_buffer(x.data)
    y_in = None
    if y is not None:
        y_in = (ctypes.c_double*len(y)).from_buffer(y.data)
    n_in = ctypes.c_int(len(x))
    return c_loghg(max_in, alpha_in, n_in, x_in, y_in, p_in, q_in,np_in, nq_in)
