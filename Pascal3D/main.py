import scipy.io
from Pascal3D import Pascal3D, Pascal3D_render, Pascal3D_all
import dataloader_utils
import matplotlib.pyplot as plt
import numpy as np
import torch

def project_points(points, angle, principal_point, camera, distance):
    f = np.prod(camera)
    C = np.zeros(3)
    a = angle[0] *np.pi/180
    e = angle[1] *np.pi/180
    th = angle[2] *np.pi/180
    C[0] = distance*np.cos(e)*np.sin(a)
    C[1] = -distance*np.cos(e)*np.cos(a)
    C[2] = distance*np.sin(e)
    a = -a
    e = e-np.pi/2

    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0,0,1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0,np.sin(e), np.cos(e)]])
    R = np.matmul(Rx, Rz)
    R2d = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    points_homo = np.ones((4, points.shape[1]))
    points_homo[:3, :] = points

    extrinsic = np.zeros((3,4))
    intrinsic = np.array([[f, 0, 0], [0, f, 0], [0,0,-1]])
    extrinsic[:,:3] = R
    extrinsic[:,3] = -np.matmul(R, C)
    P = np.matmul(intrinsic, extrinsic)

    x = np.matmul(P, points_homo)
    proj = x[:2]/x[2].reshape(1, -1)
    post_rot = np.matmul(R2d, proj)
    post_rot[1, :] = -post_rot[1, :]
    return post_rot + principal_point.reshape(2,1)


def invert_points(points2d, angle, principal_point, camera, distance, distance_obj):
    f = np.prod(camera)
    C = np.zeros(3)
    a = angle[0] *np.pi/180
    e = angle[1] *np.pi/180
    th = angle[2] *np.pi/180
    C[0] = distance*np.cos(e)*np.sin(a)
    C[1] = -distance*np.cos(e)*np.cos(a)
    C[2] = distance*np.sin(e)
    a = -a
    e = e-np.pi/2

    Rzinv = np.array([[np.cos(a), +np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0,0,1]])
    Rxinv = np.array([[1, 0, 0], [0, np.cos(e), np.sin(e)], [0,-np.sin(e), np.cos(e)]])
    Rinv = np.matmul(Rzinv, Rxinv)
    R2dinv = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])

    points = points2d - principal_point.reshape(2, 1)
    p = np.matmul(R2dinv, points)

    p[1] *= -1
    z = np.zeros((3, points2d.shape[1]))
    z[0:2,:] = -p*distance_obj/f
    z[2,:] = distance_obj
    unrot = np.matmul(Rinv, z)
    untrans = unrot + C.reshape(3,1)
    return untrans


def visualize_rotation(im, angle, principal_point, camera, distance, class_idx, cad_idx, pascal):
    ax = plt.imshow(im)
    points, faces = pascal.get_cad(class_idx, cad_idx)
    points = points.transpose()
    points_proj = project_points(points, angle, principal_point, camera, distance)
    halfsize = np.array([[im.shape[1]/2], [im.shape[0]/2]])
    points_proj = points_proj + halfsize
    plt.plot(points_proj[0], points_proj[1])

    center2d = np.array([[0.0],[0]])
    center3d = invert_points(center2d, angle, principal_point, camera, distance, distance)
    points_axis = np.zeros((3, 4))
    points_axis[:, 0] = center3d[:,0]
    points_axis[:, 1] = center3d[:,0]
    points_axis[:, 2] = center3d[:,0]
    points_axis[:, 3] = center3d[:,0]
    points_axis[:, 1:] += 0.1*np.eye(3)
    axis2d = project_points(points_axis, angle, principal_point, camera, distance)
    axis2d += halfsize
    bopp = np.zeros((2,3))
    bopp[:,0] = axis2d[:, 1] - axis2d[:, 0]
    bopp[:,1] = axis2d[:, 2] - axis2d[:, 0]
    bop[:,2] = axis2d[:, 3] - axis2d[:, 0]
    blipp = np.linalg.norm(bopp, axis=0)
    blupp = np.max(blipp)
    scale = min(im.shape[:2]) * 0.1 / blupp
    axmod = np.zeros((2,3))
    axmod[:, 0] = axis2d[:, 0] + scale * bopp[:, 0]
    axmod[:, 1] = axis2d[:, 0] + scale * bopp[:, 1]
    axmod[:, 2] = axis2d[:, 0] + scale * bopp[:, 2]
    print(axis2d)
    print(axmod)
    for i,c in zip(range(3), ['r','g','b']):
        plt.plot([axis2d[0,0], axmod[0,i]], [axis2d[1,0], axmod[1,i]], c)
    plt.show()

import tqdm
def main():
    real = Pascal3D.Pascal3D(use_warp=False)
    ds_path = 'datasets' # TODO change to where datasets are stored
    syn = Pascal3D_render.Pascal3DRendered(ds_path, 224)
    real_sampler = torch.utils.data.sampler.RandomSampler(real.get_train(), replacement=False)
    syn_size = int(0.2*len(syn))
    syn_sampler = dataloader_utils.RandomSubsetSampler(syn, syn_size)
    dataset, sampler = dataloader_utils.get_concatenated_dataset([(real.get_train(), real_sampler), (syn, syn_sampler)])
    # for idx in sampler:
    # use index 121 for visualization of warp vs non-warp
    ds = real.get_train()
    for idx in range(5502, len(ds)):
        print(idx)
        sample = ds[idx]
        image, extrinsic, class_idx, hard, intrinsic, cad_idx = sample
        print(intrinsic)
        print(extrinsic[:3,:3])
        print(extrinsic)
        image = image.transpose(1,2,0)
        points = np.array([[0.0,0,0,1],
                           [0.1,0,0,1],
                           [0,0.1,0,1],
                           [0,0,0.1,1]]).transpose()
        points = np.matmul(extrinsic, points)
        points = points/points[2]
        points = points[:3, :]
        points = np.matmul(intrinsic, points)
        plt.imshow(image)
        for p, c in zip(points[:,1:].transpose(), ['r','g','b']):
            x = [points[0,0], p[0]]
            y = [points[1,0], p[1]]
            plt.plot(x,y,c,linewidth=5)
        plt.show()
    # render cads
    for image, extrinsic, class_idx, hard, intrinsic, cad_idx in tqdm.tqdm(x.get_train(True)):
        points = np.matmul(ext, points)
        points = points/points[2]
        points = points[:3, :]
        print(points)
        points = np.matmul(intr, points)
        plt.imshow(im)
        print(points)
        for p, c in zip(points[:,1:].transpose(), ['r','g','b']):
            x = [points[0,0], p[0]]
            y = [points[1,0], p[1]]
            plt.plot(x,y,c)
        plt.show()


        nodes, _ = x.get_cad(class_idx, cad_idx)
        nodes_homo = np.ones((4, len(nodes)))
        nodes_homo[:3, :] = nodes.transpose()
        model = np.matmul(extrinsic, nodes_homo)
        model = model[:3, :]
        model /= model[2].reshape(1, -1)
        mod_proj = np.matmul(intrinsic, model)
        plt.imshow(image.transpose(1,2,0))
        plt.plot(mod_proj[0], mod_proj[1])
        plt.show()


def get_cad_index_for_class(cls):
    # temporary function allows me to interate over all cads and see that they (the cads) are aligned
    if cls == 1:
        return 8
    if cls == 2:
        return 6
    if cls == 3:
        return 6
    if cls == 4:
        return 8
    if cls == 5:
        return 6
    if cls == 6:
        return 10
    if cls == 7:
        return 10
    if cls == 8:
        return 6
    if cls == 9:
        return 5
    if cls == 10:
        return 6
    if cls == 11:
        return 4
    if cls == 12:
        return 4


if __name__ == '__main__':
    main()
