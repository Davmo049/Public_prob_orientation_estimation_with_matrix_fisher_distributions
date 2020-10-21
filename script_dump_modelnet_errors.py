import numpy
import matplotlib.pyplot as plt
from ModelNetSo3 import ModelNetSo3
import torch
from resnet import resnet101, ResnetHead
import os
import loss
import numpy as np
from PIL import Image
import tqdm
import pickle

import logger
from logger_metric import get_errors


def loss_func(net_out, R):
    A = net_out.view(-1,3,3)
    loss_v = loss.KL_Fisher(A, R)
    if loss_v is None:
        Rest = torch.unsqueeze(torch.eye(3,3, device=R.device, dtype=R.dtype), 0)
        Rest = torch.repeat_interleave(Rest, R.shape[0], 0)
        return None, Rest

    Rest = loss.batch_torch_A_to_R(A)
    return loss_v, Rest



class ListSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, l):
        self.l = l

    def __iter__(self):
        return iter(self.l)

    def __len__(self):
        return len(self.l)



def main():
    net_path = 'logs/modelnet/modelnet_int_norm'
    dump_path = 'output_data/modelnet_errors'
    dataset_location = 'datasets' # TODO update to dataset path
    device = torch.device('cpu')
    dataset = ModelNetSo3.ModelNetSo3(dataset_location)
    dataset_eval = dataset.get_eval()

    base = resnet101()
    model = ResnetHead(base, 10, 32, 512, 9)
    loggers = logger.Logger(net_path, ModelNetSo3.ModelNetSo3, load=True)
    loggers.load_network_weights(49, model, device)
    model.eval()

    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    idx = np.arange(len(dataset_eval))
    sampler = ListSampler(idx)
    dataloader = torch.utils.data.DataLoader(
            dataset_eval,
            sampler=sampler,
            batch_size=32,
            drop_last=False)

    stats = logger.SubsetLogger(None, None, 0, False) # not intended use, need to look over how to set up logger differently
    i = 0
    for batch in tqdm.tqdm(dataloader):
        # if i > 1:
        #     break
        i += 1
        image, extrinsic, class_idx_cpu, hard, intrinsic, _ = batch
        image = image.to(device)
        R = extrinsic[:, :3,:3].to(device)
        R = R.clone()
        class_idx = class_idx_cpu.to(device)
        out = model(image, class_idx)
        losses, Rest = loss_func(out, R)
        loss = torch.mean(losses)
        stats.add_samples(image, losses, out.view(-1,3,3), R, Rest, class_idx_cpu, hard)

    easy_stats, hard_stats = get_errors(stats.angular_errors, stats.class_indices, stats.hard, ModelNetSo3.ModelNetSo3Classes)

    per_class = easy_stats[1]
    for k,v in per_class.items():
        savefile = os.path.join(dump_path, '{}.txt'.format(k))
        all_errors = v[4]
        with open(savefile, 'w') as f:
            for e in all_errors:
                f.write('{}\n'.format(e))
    with open(os.path.join(dump_path, 'all.pkl'), 'wb') as f:
        pickle.dump(easy_stats, f)

    errs = per_class['bathtub'][4]
    plt.hist(errs, 100)
    plt.show()



if __name__ == '__main__':
    main()
