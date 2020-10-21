import torch
import numpy as np
import matplotlib.pyplot as plt
import torch_math
from np_math import numdiff

from geometric_utils import torch_batch_quaternion_to_rot_mat, torch_batch_euler_to_rotation
from geometric_utils import numpy_quaternion_to_rot_mat
import torch_norm_factor


epsilon_log_hg = 1e-10

def f_log_hg_torch_approx(S):
    smax = S[:,0]
    v = S[:,1:] / (smax.view(-1, 1) + epsilon_log_hg)
    vsum = torch.sum(v, dim=1)
    sndeg = 0.03125-vsum*0.02083+(0.0104+0.0209*(v[:,0]-v[:,1])**2)*vsum**2
    quot = ((3-vsum)/4)
    d = 2*sndeg / quot
    c = -quot/d
    ret = smax + c - c*torch.exp(-d*smax)
    return ret

class class_log_hg_rough_approx(torch.autograd.Function):
    # same as log_hg_torch_approx for forward, custom backward to get slightly better gradients
    @staticmethod
    def forward(ctx, S):
        smax = S[:,0]
        v = S[:,1:] / (smax.view(-1, 1) + epsilon_log_hg)
        vsum = torch.sum(v, dim=1)
        sndeg = 0.03125-vsum*0.02083+(0.0104+0.0209*(v[:,0]-v[:,1])**2)*vsum**2
        quot = ((3-vsum)/4)
        d = 2*sndeg / quot
        c = -quot/d
        ret = smax + c - c*torch.exp(-d*smax)
        ctx.save_for_backward(S)
        return ret

    @staticmethod
    def backward(ctx, grad):
        S, = ctx.saved_tensors
        tmp = torch.zeros((S.shape[0], 4), device=S.device, dtype=S.dtype)
        tmp[:, :3] = S
        tmp = tmp[:, 0].unsqueeze(1) - tmp
        weight = 1/ (1 + tmp)
        weight = weight / torch.sum(weight, dim=1).unsqueeze(1)
        weight = weight[:, :3]
        return weight * (grad).unsqueeze(1)
log_hg_torch_rough_approx = class_log_hg_rough_approx.apply

def np_log_hg_approx_backward(v):
    tmp = np.zeros(4)
    tmp[:3] = v
    tmp = np.max(tmp) - tmp
    weight = 1 / (1 + tmp)
    weight = weight / np.sum(weight)
    return weight[:3]

def np_log_hg_approx_forward(v_i):
    # note v_i sorted from smallest to largest
    smax = v_i[2]
    v = v_i[:2] / (smax + epsilon_log_hg)
    vsum = np.sum(v)
    sndeg = 0.03125-vsum*0.02083+(0.0104+0.0209*(v[0]-v[1])**2)*vsum**2
    quot = ((3-vsum)/4)
    d = 2*sndeg / quot
    c = -quot/d
    ret = smax + c - c*np.exp(-d*smax)
    return ret

class SinhExprClass(torch.autograd.Function):
    # computes x/sinh(x), custom backward/forward to get numerical stability in x==0
    @staticmethod
    def forward(ctx, input):
        exp_in = torch.exp(input)
        ctx.save_for_backward(input, exp_in)
        ret = 2*input / (exp_in - 1/exp_in)
        ret[torch.abs(input) < 0.01] = 1.0
        return ret

    def backward(ctx, grad_output):
        input, exp_in = ctx.saved_tensors
        exp_in_inv_2 = 1/(exp_in**2)
        ret = 2*(1-input-(1+input)*exp_in_inv_2)/(exp_in*(1-exp_in_inv_2)**2)
        ret[torch.abs(input) < 0.01] = 0
        return ret*grad_output
sinh_expr = SinhExprClass.apply

class LogSinhExprClass(torch.autograd.Function):
    # computes log(sinh(x)/x), custom backward/forward to get numerical stability in x==0 and for large |x|
    @staticmethod
    def forward(ctx, input):
        abs_in = torch.abs(input)
        m_exp_in_2 = torch.exp(-abs_in*2)
        ctx.save_for_backward(input, m_exp_in_2)
        ret = abs_in + torch.log((1-m_exp_in_2)/(abs_in*2))
        ret[abs_in < 0.1] = 0.0
        return ret

    def backward(ctx, grad_output):
        input, m_exp_in_2 = ctx.saved_tensors
        abs_in = torch.abs(input)
        sign_in = torch.sign(input)
        ret = 1 + m_exp_in_2 / (1-m_exp_in_2) - 1/abs_in
        ret[abs_in < 0.1] = 0.0
        return ret*grad_output*sign_in
log_sinh_expr = LogSinhExprClass.apply



def test_sinh_expr():
    x_from = -100
    x_to = 100
    steps = 10000
    steplen = (x_to-x_from)/(steps-1)
    x = np.arange(steps)
    x = x*steplen
    x = x + x_from
    xt = torch.tensor(x, requires_grad=True)
    yt = sinh_expr(xt)
    yt_np = yt.detach().numpy()
    a = torch.sum(yt)
    a.backward()
    y = x/np.sinh(x)
    y[np.abs(x) < 0.001] = 0
    plt.plot(x, y, 'g')
    plt.plot(x, yt_np, 'r--')

    plt.plot(x, y, 'g')
    plt.plot(x, yt_np, 'r--')
    plt.show()

    plt.plot(x[:-1] + steplen/2, (y[1:]-y[:-1])/steplen, 'g')
    plt.plot(x, xt.grad.numpy(), 'r--')
    plt.show()


def test_log_sinh_expr():
    x_from = -10
    x_to = 10
    steps = 1000
    steplen = (x_to-x_from)/(steps-1)
    x = np.arange(steps)
    x = x*steplen
    x = x + x_from
    xt = torch.tensor(x, requires_grad=True)
    yt = log_sinh_expr(xt)
    yt_np = yt.detach().numpy()
    a = torch.sum(yt)
    a.backward()
    y = np.log(np.sinh(x)/x)
    y[np.abs(x) < 0.001] = 0
    plt.plot(x, y, 'g')
    plt.plot(x, yt_np, 'r--')

    plt.plot(x, y, 'g')
    plt.plot(x, yt_np, 'r--')
    plt.show()

    plt.plot(x[:-1] + steplen/2, (y[1:]-y[:-1])/steplen, 'g')
    plt.plot(x, xt.grad.numpy(), 'r--')
    plt.show()


_global_svd_fail_counter = 0

def KL_approx_rough(A, R, overreg=1.05):
    global _global_svd_fail_counter
    try:
        U,S,V = torch.svd(A)
        with torch.no_grad(): # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
            rotation_candidate = torch.matmul(U,V.transpose(1,2))
            s3sign = torch.det(rotation_candidate)
        offset = s3sign*S[:, 2] - S[:, 0] - S[:, 1]
        Diag = torch.empty_like(S)
        Diag[:,0] = 2*(S[:, 0]+S[:, 1])
        Diag[:,1] = 2*(S[:, 0]-s3sign*S[:, 2])
        Diag[:,2] = 2*(S[:, 1]-s3sign*S[:, 2])
        lhg = log_hg_torch_rough_approx(Diag)
        log_norm_factor = lhg + offset
        log_exponent = -torch.matmul(A.view(-1,1,9), R.view(-1, 9,1)).view(-1)
        _global_svd_fail_counter = max(0, _global_svd_fail_counter-1)
        return log_exponent + overreg*log_norm_factor
    except RuntimeError as e:
        print(e)
        _global_svd_fail_counter += 10 # we want to allow a few failures, but not consistent ones
        if _global_svd_fail_counter > 100: # we seem to get these problems consistently
            for i in range(A.shape[0]):
                print(A[i])
            raise e
        else:
            return None



def KL_Fisher(A, R, overreg=1.05):
    # A is bx3x3
    # R is bx3x3
    global _global_svd_fail_counter
    try:
        U,S,V = torch.svd(A)
        with torch.no_grad(): # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
            rotation_candidate = torch.matmul(U,V.transpose(1,2))
            s3sign = torch.det(rotation_candidate)
        S_sign = S.clone()
        S_sign[:, 2] *= s3sign
        log_normalizer = torch_norm_factor.logC_F(S_sign)
        log_exponent = -torch.matmul(A.view(-1,1,9), R.view(-1, 9,1)).view(-1)
        _global_svd_fail_counter = max(0, _global_svd_fail_counter-1)
        return log_exponent + overreg*log_normalizer
    except RuntimeError as e:
        _global_svd_fail_counter += 10 # we want to allow a few failures, but not consistent ones
        if _global_svd_fail_counter > 100: # we seem to have gotten these problems more often than 10% of batches
            for i in range(A.shape[0]):
                print(A[i])
            raise e
        else:
            print('SVD returned NAN fail counter = {}'.format(_global_svd_fail_counter))
            return None

def KL_approx_sinh(A, R):
    # shared global variable, ugly but these two should not be used at the same time.
    global _global_svd_fail_counter
    # A is bx3x3
    # R is bx3x3
    try:
        _,S,_ = torch.svd(A)
        log_norm_factor = torch.sum(torch.sum(log_sinh_expr(S), dim=1), dim=0)
        log_exponent = -torch.matmul(A.view(-1,1,9), R.view(-1, 9,1)).view(-1)
        _global_svd_fail_counter = max(0, _global_svd_fail_counter-1)
        return log_exponent + log_norm_factor
    except RuntimeError as e:
        print(e)
        _global_svd_fail_counter += 10 # we want to allow a few failures, but not consistent ones
        if _global_svd_fail_counter > 100: # we seem to get these problems consistently
            for i in range(A.shape[0]):
                print(A[i])
            raise e
        else:
            return None


def quat_R_loss(q, R):
    Rest = torch_batch_quaternion_to_rot_mat(q)
    return torch.sum(((Rest - R)**2).view(-1,9), dim=1), Rest


def direct_quat_loss(q1, q2):
    Rest = torch_batch_quaternion_to_rot_mat(q1)
    q1 = q1 / torch.norm(q1,dim=1).view(-1,1)
    return torch.sum((q1-q2)**2, dim=1), Rest


def euler_loss(angles1, angles2):
    Rest = torch_batch_euler_to_rotation(angles1)
    return torch.sum((angles1-angles2)**2, dim=1), Rest


def batch_torch_A_to_R(A):
    U,S,V = torch.svd(A)
    with torch.no_grad(): # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
        s3sign = torch.det(torch.matmul(U,V.transpose(1,2)))
    U[:, :, 2] *= s3sign.view(-1, 1)
    R = torch.matmul(U, V.transpose(1,2))
    return R


def angle_error(t_R1, t_R2):
    ret = torch.empty((t_R1.shape[0]), dtype=t_R1.dtype, device=t_R1.device)
    rotation_offset = torch.matmul(t_R1.transpose(1,2), t_R2)
    tr_R = torch.sum(rotation_offset.view(-1,9)[:,::4], axis=1) # batch trace
    cos_angle = (tr_R - 1) / 2
    if torch.any(cos_angle < -1.1) or torch.any(cos_angle > 1.1):
        raise ValueError("angle out of range, input probably not proper rotation matrices")
    cos_angle = torch.clamp(cos_angle, -1, 1)
    angle = torch.acos(cos_angle)
    return angle*(180/np.pi)


def angle_error_np(R_1, R_2):
    tr = np.trace(np.matmul(R_1.transpose(), R_2))
    angle = np.arccos((tr-1)/2)*(180.0/np.pi)
    return angle


def numpy_hg_approx1(x,y,z):
    #model with exp as transient
    if z == 0:
        return 0.0
    v0 = x/z
    v1 = y/z
    ss = v0+v1
    sndeg = 0.03125-ss*0.02083+(0.0104+0.0209*(v0-v1)**2)*ss**2
    vs = ss+1
    d = 2*sndeg / (1-vs/4)
    c = -(1-vs/4)/d
    return z + c - c*np.exp(-d*z)

def numpy_hg_approx2(x,y,z):
    #model with c*np.log(1+exp(-d*z)) as transient
    if z == 0:
        return 0.0
    v0 = x/z
    v1 = y/z
    ss = v0+v1
    sndeg = 0.03125-ss*0.02083+(0.0104+0.0209*(v0-v1)**2)*ss**2
    vs = ss+1
    grad0 = 1-vs/4
    d = 4*sndeg/grad0
    c = 2*grad0/d
    return z + c*(np.log(1+np.exp(-d*z))-np.log(2))


class log_sinh_expr_class(torch.autograd.Function):
    # computes log(sinh(x)/x), custom backward/forward to get numerical stability in x==0 and for large |x|
    @staticmethod
    def forward(ctx, input):
        abs_in = torch.abs(input)
        m_exp_in_2 = torch.exp(-abs_in*2)
        ctx.save_for_backward(input, m_exp_in_2)
        ret = abs_in + torch.log((1-m_exp_in_2)/(abs_in*2))
        ret[abs_in < 0.1] = 0.0
        return ret

    def backward(ctx, grad_output):
        input, m_exp_in_2 = ctx.saved_tensors
        abs_in = torch.abs(input)
        sign_in = torch.sign(input)
        ret = 1 + m_exp_in_2 / (1-m_exp_in_2) - 1/abs_in
        ret[abs_in < 0.1] = 0.0
        return ret*grad_output*sign_in

log_sinh_expr = log_sinh_expr_class.apply
