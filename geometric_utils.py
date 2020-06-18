import numpy as np
import torch


def numpy_quaternion_to_rot_mat(q):
    q /= np.linalg.norm(q)
    qq = 2*q[:3].reshape(1, -1)*q[:3].reshape(-1, 1)
    qq[0, 1] -= 2*q[3]*q[2]
    qq[0, 2] += 2*q[3]*q[1]
    qq[1, 2] -= 2*q[3]*q[0]
    qq[2, 1] += 2*q[3]*q[0]
    qq[2, 0] -= 2*q[3]*q[1]
    qq[1, 0] += 2*q[3]*q[2]
    q2 = q**2
    qq[0,0] = 1 - 2*(q2[1]+ q2[2])
    qq[1,1] = 1 - 2*(q2[0]+ q2[2])
    qq[2,2] = 1 - 2*(q2[0]+ q2[1])
    return qq


def torch_batch_quaternion_to_rot_mat(q_unnorm):
    # q is (-1, 4)
    q = q_unnorm / torch.norm(q_unnorm, dim=1).view(-1, 1)
    qq = torch.empty((q.shape[0], 3,3), dtype=q.dtype, device=q.device)
    qq[:, 0, 1] = 2*(q[:, 0]*q[:,1] - q[:, 3]*q[:, 2])
    qq[:, 0, 2] = 2*(q[:, 0]*q[:,2] + q[:, 3]*q[:, 1])
    qq[:, 1, 2] = 2*(q[:, 1]*q[:,2] - q[:, 3]*q[:, 0])
    qq[:, 2, 1] = 2*(q[:, 1]*q[:,2] + q[:, 3]*q[:, 0])
    qq[:, 2, 0] = 2*(q[:, 0]*q[:,2] - q[:, 3]*q[:, 1])
    qq[:, 1, 0] = 2*(q[:, 0]*q[:,1] + q[:, 3]*q[:, 2])
    q2 = q[:,:3]**2
    qq[:, 0,0] = 1 - 2*(q2[:, 1] + q2[:, 2])
    qq[:, 1,1] = 1 - 2*(q2[:, 0] + q2[:, 2])
    qq[:, 2,2] = 1 - 2*(q2[:, 0] + q2[:, 1])
    return qq

def numpy_euler_to_rotation(angles):
    # note angles are z,y,x intrinsic
    ax = angles[0]
    cx = np.cos(ax)
    sx = np.sin(ax)
    ay = angles[1]
    cy = np.cos(ay)
    sy = np.sin(ay)
    az = angles[2]
    cz = np.cos(az)
    sz = np.sin(az)
    Ax = np.array([[1,0,0], [0, cx, -sx], [0, sx, cx]])
    Ay = np.array([[cy,0,sy], [0, 1, 0], [-sy, 0, cy]])
    Az = np.array([[cz, -sz,0], [sz, cz, 0], [0, 0, 1]])
    R = np.matmul(Az, np.matmul(Ay, Ax))
    return R

def numpy_quaternion_to_angles(q):
    # note angles are z,y,x intrinsic
    q /= np.linalg.norm(q)
    roll = np.arctan2(2*(q[3]*q[0]+q[1]*q[2]), 1-2*(q[0]**2+q[1]**2))
    pitch = np.arcsin(2*(q[3]*q[1]-q[0]*q[2]))
    yaw = np.arctan2(2*(q[3]*q[2]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2))
    return np.array([roll, pitch, yaw])

def torch_batch_euler_to_rotation(angles):
    # note angles are z,y,x intrinsic
    ax = angles[:, 0]
    cx = torch.cos(ax)
    sx = torch.sin(ax)
    ay = angles[:, 1]
    cy = torch.cos(ay)
    sy = torch.sin(ay)
    az = angles[:, 2]
    cz = torch.cos(az)
    sz = torch.sin(az)
    Ax = torch.zeros((angles.shape[0], 3,3), dtype=angles.dtype, device=angles.device)
    Ax[:, 0,0]=1
    Ax[:, 1,1]=cx
    Ax[:, 1,2]=-sx
    Ax[:, 2,1]=sx
    Ax[:, 2,2]=cx
    Ay = torch.zeros((angles.shape[0], 3,3), dtype=angles.dtype, device=angles.device)
    Ay[:, 1,1]=1
    Ay[:, 0,0]=cy
    Ay[:, 0,2]=sy
    Ay[:, 2,0]=-sy
    Ay[:, 2,2]=cy
    Az = torch.zeros((angles.shape[0], 3,3), dtype=angles.dtype, device=angles.device)
    Az[:, 2,2]=1
    Az[:, 0,0]=cz
    Az[:, 0,1]=-sz
    Az[:, 1,0]=sz
    Az[:, 1,1]=cz
    ret = torch.matmul(Az, torch.matmul(Ay, Ax))
    return ret

def numpy_aet_to_rot(aet):
    a = aet[0]
    e = aet[1]
    t = aet[2]
    Ra = np.array([[np.cos(a), -np.sin(a), 0],
                   [np.sin(a), np.cos(a), 0],
                   [0,0,1]])
    Re = np.array([[np.cos(e), 0,         -np.sin(e)],
                   [0,         1,         0],
                   [np.sin(a), 0,         np.cos(e)]])
    Rt = np.array([[np.cos(t), -np.sin(t), 0],
                   [np.sin(t), np.cos(t), 0],
                   [0,0,1]])
    R = np.matmul(Rt, np.matmul(Re, Ra))
    return R
