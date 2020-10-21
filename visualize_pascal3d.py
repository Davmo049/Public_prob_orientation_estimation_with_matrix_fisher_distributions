import numpy
import matplotlib.pyplot as plt
from Pascal3D import Pascal3D
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
    points = np.array([[0.0,0,0,1],
                       [0.3,0,0,1],
                       [0,0.3,0,1],
                       [0,0,0.3,1]]).transpose()
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


def get_cpu_stats():
    net_path = 'logs/pascal/pascal_new_norm_full'
    dataset_location = 'datasets' # TODO update with dataset path
    device = torch.device('cpu')
    dataset = Pascal3D.Pascal3D(train_all=True)
    dataset_vis = dataset.get_eval()

    base = resnet101()
    model = ResnetHead(base, 13, 32, 512, 9)

    loggers = logger.Logger(net_path, Pascal3D.PascalClasses, load=True)
    loggers.load_network_weights(119, model, device)
    model.eval()
    np.random.seed(9001)
    dataloader = torch.utils.data.DataLoader(
            dataset_vis,
            shuffle=True,
            batch_size=1,
            drop_last=False)
    err_per_class = {}
    print(len(dataloader))
    for i, batch in enumerate(dataloader):
        if i % 10 == 0:
            print(i)
        image, extrinsic, class_idx_cpu, hard, intrinsic, _ = batch
        extrinsic_np = extrinsic[0].numpy()
        intrinsic_np = intrinsic[0].numpy()
        im_np = image[0].numpy().transpose(1,2,0)
        R_gt = extrinsic[0, :3,:3].numpy()
        out = model(image, class_idx_cpu).view(-1,3,3)
        R_est = loss.batch_torch_A_to_R(out).detach().cpu().view(3,3).numpy()
        err = loss.angle_error_np(R_gt, R_est)
        k = class_idx_cpu.numpy()[0]
        errs = err_per_class.get(k, [])
        errs.append(err)
        err_per_class[k] = errs

    for k in range(13):
        if k in err_per_class:
            print(k)
            print(np.median(err_per_class[k]))





def visualize_random_errors():
    net_path = 'logs/pascal/pascal_new_norm_full'
    image_dir_out = 'plots/random_errors'
    dataset_location = 'datasets' # TODO update with dataset path
    device = torch.device('cpu')
    dataset = Pascal3D.Pascal3D(train_all=True)
    dataset_vis = dataset.get_eval()

    base = resnet101()
    model = ResnetHead(base, 13, 32, 512, 9)
    loggers = logger.Logger(net_path, Pascal3D.PascalClasses, load=True)
    loggers.load_network_weights(119, model, device)
    model.eval()

    if not os.path.exists(image_dir_out):
        os.makedirs(image_dir_out)
    np.random.seed(9001)
    idx = np.arange(len(dataset_vis))
    np.random.shuffle(idx)
    sampler = ListSampler(idx)
    dataloader = torch.utils.data.DataLoader(
            dataset_vis,
            sampler=sampler,
            batch_size=1,
            drop_last=False)

    fig, axs = plt.subplots(3,4, figsize=(10,7.5), dpi=100*4)
    for i, batch in enumerate(dataloader):
        if i == 12:
            break
        ax = axs[i//4][i%4]
        image, extrinsic, class_idx_cpu, hard, intrinsic, _ = batch
        extrinsic_np = extrinsic[0].numpy()
        intrinsic_np = intrinsic[0].numpy()
        im_np = image[0].numpy().transpose(1,2,0)
        R_gt = extrinsic[0, :3,:3].numpy()
        out = model(image, class_idx_cpu).view(-1,3,3)
        R_est = loss.batch_torch_A_to_R(out).detach().cpu().view(3,3).numpy()
        err = loss.angle_error_np(R_gt, R_est)

        extr_est = np.copy(extrinsic_np)
        extr_est[:3,:3] = R_est
        points_est = proj_axis(extr_est, intrinsic_np)
        points_true = proj_axis(extrinsic_np, intrinsic_np)


        ax.imshow(im_np)
        for p, c in zip(points_est[:,1:].transpose(), ['r','g','b']):
            x = [points_est[0,0], p[0]]
            y = [points_est[1,0], p[1]]
            ax.plot(x,y,c,linewidth=5)

        for p, c in zip(points_true[:,1:].transpose(), ['m','y','c']):
            x = [points_true[0,0], p[0]]
            y = [points_true[1,0], p[1]]
            ax.plot(x,y,c,linewidth=2)
        ax.title.set_text("error: {:3.2f}".format(err))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    image_out = os.path.join(image_dir_out, 'im_errors.pdf')
    plt.savefig(image_out)


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



def go_through_visualizations():
    net_path = 'logs/pascal/pascal_new_norm_full'
    dataset_location = 'datasets' # TODO update with dataset path
    device = torch.device('cpu')
    dataset = Pascal3D.Pascal3D(train_all=True)
    dataset_vis = dataset.get_eval()

    base = resnet101()
    model = ResnetHead(base, 13, 32, 512, 9)
    loggers = logger.Logger(net_path, Pascal3D.PascalClasses, load=True)
    loggers.load_network_weights(119, model, device)
    model.eval()

    np.random.seed(9001)
    idx = np.arange(1100,len(dataset_vis), 1)
    sampler = ListSampler(idx)
    dataloader = torch.utils.data.DataLoader(
            dataset_vis,
            sampler=sampler,
            batch_size=1,
            drop_last=False)

    im_weight = 4
    for i, (idx, batch) in enumerate(zip(idx, dataloader)):
        print(idx)
        fig = plt.figure(figsize=(10,3), dpi=100*4)
        gs = fig.add_gridspec(im_weight+3, 1)
        im_ax = fig.add_subplot(gs[:im_weight, 0])
        image, extrinsic, class_idx_cpu, hard, intrinsic, _ = batch
        extrinsic_np = extrinsic[0].numpy()
        intrinsic_np = intrinsic[0].numpy()
        im_np = image[0].numpy().transpose(1,2,0)
        R_gt = extrinsic[0, :3,:3].numpy()
        out = model(image, class_idx_cpu).view(-1,3,3)
        R_est = loss.batch_torch_A_to_R(out).detach().cpu().view(3,3).numpy()
        err = loss.angle_error_np(R_gt, R_est)

        F = out[0].detach().numpy()
        extr_est = np.copy(extrinsic_np)
        extr_est[:3,:3] = R_est

        points_est = proj_axis(extr_est, intrinsic_np)
        points_true = proj_axis(extrinsic_np, intrinsic_np)


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
            ax = fig.add_subplot(gs[rot_axis+im_weight, 0])
            ax.plot(x,y, c)
            ax.plot([ml, ml], [0.0, 1.0], 'k')
        plt.show()
 

def visualize_probs():
    net_path = 'logs/pascal/pascal_new_norm_full'
    image_dir_out = 'plots/probs'
    dataset_location = 'datasets' # TODO update with dataset path
    device = torch.device('cpu')
    dataset = Pascal3D.Pascal3D(train_all=True)
    dataset_vis = dataset.get_eval()

    base = resnet101()
    model = ResnetHead(base, 13, 32, 512, 9)
    loggers = logger.Logger(net_path, Pascal3D.PascalClasses, load=True)
    loggers.load_network_weights(119, model, device)
    model.eval()

    if not os.path.exists(image_dir_out):
        os.makedirs(image_dir_out)

    np.random.seed(9001)
    # idx = np.arange(1000,len(dataset_vis), 1)
    idx = [4008, 1126, 9024, 11159]
    sampler = ListSampler(idx)
    dataloader = torch.utils.data.DataLoader(
            dataset_vis,
            sampler=sampler,
            batch_size=1,
            drop_last=False)

    im_weight = 4
    fig = plt.figure(figsize=(10,3), dpi=100*4)
    gs = fig.add_gridspec(im_weight+3, 4)
    for i, (idx, batch) in enumerate(zip(idx, dataloader)):
        print(i)
        if i == 4:
           break
        im_ax = fig.add_subplot(gs[:im_weight, i])
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
            ax = fig.add_subplot(gs[rot_axis+im_weight, i])
            ax.plot(x,y, c)
            ax.plot([ml, ml], [0.0, 1.0], 'k')
    plt.show()
    # image_out = os.path.join(image_dir_out, 'im_prob.pdf')
    # plt.savefig(image_out)



def main():
    # go_through_visualizations()
    visualize_random_errors()
    visualize_probs()
    # get_cpu_stats()


if __name__ == '__main__':
    main()
