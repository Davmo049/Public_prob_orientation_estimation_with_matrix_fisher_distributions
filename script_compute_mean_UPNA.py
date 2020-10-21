import numpy
import matplotlib.pyplot as plt
import torch
from resnet import resnet101, ResnetHead
from Pascal3D import Pascal3D
from ModelNetSo3 import ModelNetSo3
import logger
import os
import loss
import numpy as np
import pickle
from UPNA import UPNA

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
    points = points/points[2]
    points = points[:3, :]
    points = np.matmul(intrinsic, points)
    return points[:2, :]




def get_prob(rot_axis, FR):
    print(FR)
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
    net_path = 'logs/upna/upna_int_norm'
    image_dir_out = 'plots/probs'
    dataset_location = 'datasets' # TODO update to dataset path
    device = torch.device('cpu')
    dataset = UPNA.UPNA(dataset_location)
    dataset_vis = dataset.get_eval()

    base = resnet101()
    model = ResnetHead(base, 1, 0, 512, 9)
    loggers = logger.Logger(net_path, ModelNetSo3.ModelNetSo3Classes, load=True)
    loggers.load_network_weights(119, model, device)
    model.eval()

    if not os.path.exists(image_dir_out):
        os.makedirs(image_dir_out)
    np.random.seed(29001)
    idx = np.arange(len(dataset_vis))
    np.random.shuffle(idx)
    # idx = [2822, 8446, 6171, 3479]
    sampler = ListSampler(idx)
    dataloader = torch.utils.data.DataLoader(
            dataset_vis,
            sampler=sampler,
            batch_size=1,
            drop_last=False)

    errors = []
    load_pkl = False
    if load_pkl and os.path.exists('UPNA_errors.pkl'):
        with open('UPNA_errors.pkl', 'rb') as f:
            errors = pickle.load(f)
    else:
        print(len(dataloader))
        for i, (idx, batch) in enumerate(zip(idx, dataloader)):
            print(i)
            image, extrinsic, class_idx_cpu, hard, intrinsic, _ = batch
            extrinsic_np = extrinsic[0].numpy()
            intrinsic_np = intrinsic[0].numpy()
            im_np = image[0].numpy().transpose(1,2,0)
            R_gt = extrinsic[0, :3,:3].numpy()
            out = model(image, class_idx_cpu).view(-1,3,3)
            R_est = loss.batch_torch_A_to_R(out).detach().cpu().view(3,3).numpy()
            err = loss.angle_error_np(R_gt, R_est)
            errors.append(err)
            print(err)
        with open('UPNA_errors.pkl', 'wb') as f:
            pickle.dump(errors, f)
    print(np.mean(errors))
    print(np.median(errors))
    plt.hist(errors,100)
    plt.show()


def main():
    visualize_probs()


if __name__ == '__main__':
    main()
