import logger_metric

import logger
import tensorboardX
import loss
import matplotlib.pyplot as plt

from logger_metric import get_errors, print_stats
import numpy as np
import torch
import os
import io
from PIL import Image
import pickle
import json

class Logger():
    def __init__(self, logger_path, class_enum, config=None, load=False):
        self.logger_path = logger_path
        if not load and os.path.exists(logger_path):
            raise Exception("rerunning old training")
        elif not os.path.exists(logger_path):
            os.makedirs(logger_path)
            os.makedirs(os.path.join(logger_path, 'saved_weights'))
            if config is not None:
                with open(os.path.join(logger_path, 'config.json'), 'w') as f:
                    json.dump(config.json_serialize(), f)
        self.class_enum = class_enum
        self.tb_summary_writer = tensorboardX.SummaryWriter(os.path.join(self.logger_path, 'logging'))

    def get_train_logger(self, epoch, verbose=False):
        return SubsetLogger(self, 'train', epoch, verbose)

    def get_validation_logger(self, epoch, verbose=False):
        return SubsetLogger(self, 'validation', epoch, verbose)

    def save_network(self, epoch, model):
        path = os.path.join(self.logger_path, 'saved_weights', 'state_dict_{}.pkl'.format(epoch))
        torch.save(model.state_dict(), path)

    def load_network_weights(self, epoch, model, device):
        path = os.path.join(self.logger_path, 'saved_weights', 'state_dict_{}.pkl'.format(epoch))
        with open(path, 'rb') as f:
            state_dict = torch.load(f, map_location=device)
        model.load_state_dict(state_dict)

class SubsetLogger():
    def __init__(self, logger, subset_name, epoch, verbose):
        self.logger = logger
        self.subset_name = subset_name
        self.epoch = epoch
        self.verbose = verbose
        self.angular_errors = []
        self.class_indices = []
        self.hard = []
        self.sum_loss = 0.0
        self.num_samples_loss = 0

    def add_samples(self, images, losses, prob_params, R_gt, R_est, class_idx, hard):
        # all inputs are tensors on training device
        current_idx = len(self.angular_errors)
        ang_err = loss.angle_error(R_est, R_gt).cpu().detach().numpy()
        self.angular_errors += list(ang_err)
        self.class_indices += list(class_idx.detach().cpu().numpy())
        self.hard += list(hard.cpu().numpy())
        self.sum_loss += torch.sum(losses).detach().cpu().numpy()
        self.num_samples_loss += losses.shape[0]

    def finish(self):
        tb_writer = self.logger.tb_summary_writer
        print('epoch {}: loss {}'.format(self.epoch, float(self.sum_loss/self.num_samples_loss)))
        tb_writer.add_scalar('{}/loss'.format(self.subset_name), float(self.sum_loss/self.num_samples_loss), self.epoch)
        easy_stats, all_stats = get_errors(self.angular_errors, self.class_indices, self.hard, self.logger.class_enum)
        x = [(all_stats, 'all')]
        if np.any(self.hard):
            x.append([easy_stats, 'easy'])
        for y in x:
            stats = y[0]
            stat_name = y[1]
            stats_global = stats[0]
            stats_per_class = stats[1]
            tb_writer.add_scalar('{}/Median_{}'.format(self.subset_name, stat_name), stats_global[0], self.epoch)
            tb_writer.add_scalar('{}/Acc_at_30_{}'.format(self.subset_name, stat_name), stats_global[1], self.epoch)
            tb_writer.add_scalar('{}/Acc_at_15_{}'.format(self.subset_name, stat_name), stats_global[2], self.epoch)
            tb_writer.add_scalar('{}/Acc_at_7_5_{}'.format(self.subset_name, stat_name), stats_global[3], self.epoch)
            tb_writer.add_histogram('{}/angle_errors_{}'.format(self.subset_name, stat_name), np.array(stats_global[4]), self.epoch)
            for class_name, class_stats in stats_per_class.items():
                tb_writer.add_scalar('per_class_{}/{}_Median_{}'.format(self.subset_name, class_name, stat_name), class_stats[0], self.epoch)
                tb_writer.add_scalar('per_class_{}/{}_Acc_at_30_{}'.format(self.subset_name, class_name, stat_name), class_stats[1], self.epoch)
                tb_writer.add_scalar('per_class_{}/{}_Acc_at_15_{}'.format(self.subset_name, class_name, stat_name), class_stats[2], self.epoch)
                tb_writer.add_scalar('per_class_{}/{}_Acc_at_7_5_{}'.format(self.subset_name, class_name, stat_name), class_stats[3], self.epoch)
                tb_writer.add_histogram('per_class_{}/{}_angle_errors_{}'.format(self.subset_name, class_name, stat_name), np.array(class_stats[4]), self.epoch)



def generate_axis_plot(image, R, title='', confidence=None):
    fig = plt.figure()
    if title != '':
        plt.title(title)
    plt.imshow(image)
    extrinsic = np.eye(4)
    extrinsic[:3,:3] = R
    extrinsic[:3, 3] = np.array([0,0,10])
    extrinsic[3, 3] = 1
    im_sz = image.shape[0]
    intrinsic = np.array([[im_sz/2, 0,       im_sz/2],
                          [0,       im_sz/2, im_sz/2],
                          [0,       0,       1]])
    if confidence is None:
        confidence = np.ones((3))
    nodes = np.array([[0.0,0,0,1],
                      [confidence[0],0,0,1],
                      [0,confidence[1],0,1],
                      [0,0,confidence[2],1]]).transpose()
    nodes = np.matmul(extrinsic, nodes)
    nodes = nodes[:3, :]
    nodes /= nodes[2].reshape(1, -1)
    nodes = np.matmul(intrinsic, nodes)
    for i,c in enumerate(['r','g','b']):
        x = [nodes[0,0], nodes[0, i+1]]
        y = [nodes[1,0], nodes[1, i+1]]
        plt.plot(x, y, c)

    return fig
