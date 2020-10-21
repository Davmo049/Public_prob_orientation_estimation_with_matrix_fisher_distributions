import numpy
import matplotlib.pyplot as plt
from ModelNetSo3 import ModelNetSo3
import torch
from resnet import resnet101, ResnetHead
import logger
import os
import loss
import numpy as np
from PIL import Image

class ListSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, l):
        self.l = l

    def __iter__(self):
        return iter(self.l)

    def __len__(self):
        return len(self.l)


def proj_axis(extrinsic, intrinsic):
    intrinsic = np.copy(intrinsic)
    intrinsic[:2, :2] *= 2
    d = 1
    points = np.array([[0.0,0,0,1],
                       [d,0,0,1],
                       [0,d,0,1],
                       [0,0,d,1]]).transpose()
    points = np.matmul(extrinsic, points)
    points = points/points[2].reshape(1,4)
    points = points[:3, :]
    points = np.matmul(intrinsic, points)
    points = points[:2, :]
    points_diff = points[:, 1:] - points[:, 0].reshape(2,1)
    box_min = - points[:, 0]
    box_max = 213 - points[:, 0]
    scale_required = 1.0
    for i in range(3):
        diff = points_diff[:, i]
        if diff[0] > box_max[0]:
            scale_required = min(scale_required, box_max[0]/diff[0])
        if diff[0] < box_min[0]:
            scale_required = min(scale_required, box_min[0]/diff[0])
        if diff[1] > box_max[1]:
            scale_required = min(scale_required, box_max[1]/diff[1])
        if diff[1] < box_min[1]:
            scale_required = min(scale_required, box_min[1]/diff[1])

    points[:,1:] = points_diff * scale_required + points[:, 0].reshape(2,1)
    return points


def get_prob(rot_axis, FR):
    a = np.trace(FR)-FR[rot_axis,rot_axis]
    x = np.arange(-np.pi, np.pi, 2*np.pi/1000)
    y = np.exp(a*(np.cos(x)-1))
    return x, y


def get_axis_max_likelihood(rot_axis, mat):
    i1, i2 = list(set(range(3))-set([rot_axis]))
    c = mat[i1,i1]+mat[i2,i2]
    s = mat[i1,i2]-mat[i2,i1]
    theta = numpy.arccos(c/np.sqrt(c**2+s**2))
    return theta



def visualize_probs():
    net_path = 'logs/modelnet/modelnet_int_norm'
    image_dir_out = 'plots/modelnet_probs'
    dataset_location = 'datasets' # TODO update with dataset path
    device = torch.device('cpu')
    dataset = ModelNetSo3.ModelNetSo3()
    dataset_vis = dataset.get_eval()

    base = resnet101()
    model = ResnetHead(base, 10, 32, 512, 9)
    loggers = logger.Logger(net_path, ModelNetSo3.ModelNetSo3Classes, load=True)
    loggers.load_network_weights(49, model, device)
    model.eval()

    if not os.path.exists(image_dir_out):
        os.makedirs(image_dir_out)
    np.random.seed(9001)
    # idx = np.arange(14450,len(dataset_vis), 100)
    idx = [15254]
    sampler = ListSampler(idx)
    dataloader = torch.utils.data.DataLoader(
            dataset_vis,
            sampler=sampler,
            batch_size=1,
            drop_last=False)

    # fig = plt.figure(figsize=(10,3), dpi=100*4)
    im_weight = 4
    # gs = fig.add_gridspec(im_weight+3, 4, hspace=0.5)
    epochs = [0, 20, 40, 49]
    for i, (idx, batch) in enumerate(zip(idx, dataloader)):
        for ep in epochs:
            print(ep)
            loggers.load_network_weights(ep, model, device)

            fig = plt.figure(figsize=(10,3), dpi=100*4)
            gs = fig.add_gridspec(im_weight+3, 1, hspace=0.5)
            im_ax = fig.add_subplot(gs[:im_weight])
            image, extrinsic, class_idx_cpu, hard, intrinsic, _ = batch
            extrinsic_np = extrinsic[0].numpy()
            intrinsic_np = intrinsic[0].numpy()
            im_np = image[0].numpy().transpose(1,2,0)
            R_gt = extrinsic[0, :3,:3].numpy()
            out = model(image, class_idx_cpu).view(-1,3,3)
            R_est = loss.batch_torch_A_to_R(out).detach().cpu().view(3,3).numpy()
            j = Image.fromarray((im_np*255).astype(np.uint8))
            image_out = os.path.join(image_dir_out, 'im_prob_{}.png'.format(i))
            j.save(image_out)
            print("F")
            print(out.detach().numpy())
            print("gt")
            print(R_gt)
            print("est")
            print(R_est)
            err = loss.angle_error_np(R_gt, R_est)
    
            F = out[0].detach().numpy()
            extr_est = np.copy(extrinsic_np)
            print(extr_est)
            extr_est[:3,:3] = R_est
    
            points_est = proj_axis(extr_est, intrinsic_np)
            print("points_est")
            print(points_est)
            points_true = proj_axis(extrinsic_np, intrinsic_np)
            print("points_true")
            print(points_true)
    
    
            im_ax.imshow(im_np)
            for p, c in zip(points_est[:,1:].transpose(), ['r','g','b']):
                x = [points_est[0,0], p[0]]
                y = [points_est[1,0], p[1]]
                im_ax.plot(x,y,c,linewidth=5)
    
            for p, c in zip(points_true[:,1:].transpose(), ['m','y','c']):
                x = [points_true[0,0], p[0]]
                y = [points_true[1,0], p[1]]
                im_ax.plot(x,y,c,linewidth=2)
            im_ax.axes.get_xaxis().set_visible(False)
            im_ax.axes.get_yaxis().set_visible(False)
            for rot_axis, c in zip(range(3), ['r', 'g', 'b']):
                x, y = get_prob(rot_axis, np.matmul(F.transpose(), R_est))
                ml = get_axis_max_likelihood(rot_axis, np.matmul(F.transpose(), R_gt))
                ax = fig.add_subplot(gs[rot_axis+im_weight])
                ax.plot(x,y, c)
                ax.plot([ml, ml], [0.0, 1.0], 'k')
    
            plt.show()
    # image_out = os.path.join(image_dir_out, 'im_prob.pdf')
    # plt.savefig(image_out)



def main():
    # visualize_random_errors()
    visualize_probs()


if __name__ == '__main__':
    main()
