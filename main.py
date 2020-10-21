import matplotlib.pyplot as plt
import numpy as np
import tqdm
from Pascal3D import Pascal3D, Pascal3D_render, Pascal3D_all
from ModelNetSo3 import ModelNetSo3
from resnet import resnet50, resnet101, ResnetHead
from UPNA import UPNA
import loss
import torch
import os
import tqdm
import argparse
import dataloader_utils
import logger
import matplotlib
import json
matplotlib.use('Agg')

dataset_dir = 'datasets' # TODO change with dataset path

def vmf_loss(net_out, R, overreg=1.05):
    A = net_out.view(-1,3,3)
    loss_v = loss.KL_Fisher(A, R, overreg=overreg)
    if loss_v is None:
        Rest = torch.unsqueeze(torch.eye(3,3, device=R.device, dtype=R.dtype), 0)
        Rest = torch.repeat_interleave(Rest, R.shape[0], 0)
        return None, Rest

    Rest = loss.batch_torch_A_to_R(A)
    return loss_v, Rest

def get_pascal_no_warp_loaders(batch_size, train_all, voc_train):
    dataset = Pascal3D.Pascal3D(dataset_dir, train_all=train_all, use_warp=False, voc_train=voc_train)
    dataloader_train = torch.utils.data.DataLoader(
        dataset.get_train(False),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=True)
    dataloader_eval = torch.utils.data.DataLoader(
        dataset.get_eval(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=False)
    return dataloader_train, dataloader_eval


def get_pascal_loaders(batch_size, train_all, use_synthetic_data, use_augment, voc_train):
    if use_synthetic_data:
        return get_pascal_synthetic(batch_size, train_all, use_augment, voc_train)
    else:
        dataset = Pascal3D.Pascal3D(dataset_dir, train_all=train_all, use_warp=True, voc_train=voc_train)
        dataloader_train = torch.utils.data.DataLoader(
            dataset.get_train(use_augment),
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
            pin_memory=True,
            drop_last=True)
        dataloader_eval = torch.utils.data.DataLoader(
            dataset.get_eval(),
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
            pin_memory=True,
            drop_last=False)
        return dataloader_train, dataloader_eval


def get_pascal_synthetic(batch_size, train_all, use_augmentation, voc_train):
    dataset_real = Pascal3D.Pascal3D(dataset_dir, train_all=train_all, use_warp=True, voc_train=voc_train)
    train_real = dataset_real.get_train(use_augmentation)
    real_sampler = torch.utils.data.sampler.RandomSampler(train_real, replacement=False)
    dataset_rendered = Pascal3D_render.Pascal3DRendered(dataset_dir)
    rendered_size = int(0.2*len(dataset_rendered)) # use 20% of synthetic data for training per epoch
    rendered_sampler = dataloader_utils.RandomSubsetSampler(dataset_rendered, rendered_size)
    dataset_train, sampler_train = dataloader_utils.get_concatenated_dataset([(train_real, real_sampler), (dataset_rendered, rendered_sampler)])

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=batch_size,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=True)

    dataloader_eval = torch.utils.data.DataLoader(
        dataset_real.get_eval(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=False)
    return dataloader_train, dataloader_eval

def get_upna_loaders(batch_size, train_all):
    dataset = UPNA.UPNA(dataset_dir)
    train_ds = dataset.get_train()
    dataloader_train = torch.utils.data.DataLoader(
        dataset.get_train(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=True)

    dataloader_eval = torch.utils.data.DataLoader(
        dataset.get_eval(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=False)
    return dataloader_train, dataloader_eval

def get_modelnet_loaders(batch_size, train_all):
    dataset = ModelNetSo3.ModelNetSo3(dataset_dir)
    dataloader_train = torch.utils.data.DataLoader(
        dataset.get_train(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0, # I suspect the suprocess fails to free their transactions when terminating? not too much processing done in dataloader anyways
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=True)
    dataloader_eval = torch.utils.data.DataLoader(
        dataset.get_eval(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0, # I suspect the suprocess fails to free their transactions when terminating? not too much processing done in dataloader anyways
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=False)
    return dataloader_train, dataloader_eval

def train_model(loss_func, out_dim, train_setting):
    # device = 'cpu'
    device = 'cuda'
    batch_size = 32
    train_all = True # train_all=False when decisions were made
    config = train_setting.config
    run_name = train_setting.run_name
    base = resnet101(pretrained=True, progress=True)
    if config.type == 'pascal':
        num_classes=12+1 # +1 due to one indexed classes
    elif config.type == 'modelnet':
        num_classes=10
    elif config.type == 'upna':
        num_classes=1
    model = ResnetHead(base, num_classes, config.embedding_dim, 512, out_dim)
    model.to(device)

    if config.type == 'pascal':
        use_synthetic_data = config.synthetic_data
        use_augmentation = config.data_aug
        use_warp = config.warp
        voc_train = config.pascal_train
        if not use_warp:
            assert(not use_synthetic_data)
            assert(not use_augmentation)
            dataloader_train, dataloader_eval = get_pascal_no_warp_loaders(batch_size, train_all, voc_train)
        else:
            dataloader_train, dataloader_eval = get_pascal_loaders(batch_size, train_all, use_synthetic_data, use_augmentation, voc_train)
    elif config.type == 'modelnet':
        dataloader_train, dataloader_eval = get_modelnet_loaders(batch_size, train_all)
    elif config.type == 'upna':
        dataloader_train, dataloader_eval = get_upna_loaders(batch_size, train_all)
    else:
        raise Exception("Unknown config: {}".config.format())

    if model.class_embedding is None:
        finetune_parameters = model.head.parameters()
    else:
        finetune_parameters = list(model.head.parameters()) + list(model.class_embedding.parameters())

    if config.type == 'modelnet':
        num_epochs = 50
        drop_epochs = [30, 40, 45, np.inf]
        stop_finetune_epoch = 2
    else:
        num_epochs = 120
        drop_epochs = [30, 60, 90, np.inf]
        stop_finetune_epoch = 3
    drop_idx = 0

    cur_lr = 0.01
    opt = torch.optim.SGD(finetune_parameters, lr=cur_lr)
    if config.type == 'pascal':
        class_enum = Pascal3D.PascalClasses
    else:
        class_enum = ModelNetSo3.ModelNetSo3Classes
    log_dir = 'logs/{}/{}'.format(config.type, run_name)
    loggers = logger.Logger(log_dir, class_enum, config=config)
    for epoch in range(num_epochs):
        verbose = epoch % 20 == 0 or epoch == num_epochs-1
        if epoch == drop_epochs[drop_idx]:
            cur_lr *= 0.1
            drop_idx += 1
            opt = torch.optim.SGD(model.parameters(), lr=cur_lr)
        elif epoch == stop_finetune_epoch:
            opt = torch.optim.SGD(model.parameters(), lr=cur_lr)
        logger_train = loggers.get_train_logger(epoch, verbose)
        model.train()
        for image, extrinsic, class_idx_cpu, hard, _, _ in dataloader_train:
            image = image.to(device)
            R = extrinsic[:, :3,:3].to(device)
            class_idx = class_idx_cpu.to(device)
            out = model(image, class_idx)
            losses, Rest = loss_func(out, R, overreg=1.025)

            if losses is not None:
                loss = torch.mean(losses)
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                losses = torch.zeros(R.shape[0], dtype=R.dtype, device=R.device)
            logger_train.add_samples(image, losses, out.view(-1,3,3), R, Rest, class_idx_cpu, hard)
        logger_train.finish()
        logger_train = None
        image=None
        R=None
        class_idx = None
        out = None
        loss_value=None
        Rest=None

        logger_eval = loggers.get_validation_logger(epoch, verbose)
        model.eval()
        with torch.no_grad():
            for image, extrinsic, class_idx_cpu, hard, _, _ in dataloader_eval:
                image = image.to(device)
                R = extrinsic[:,:3,:3].to(device)
                class_idx = class_idx_cpu.to(device)
                out = model(image, class_idx)
                losses, Rest = loss_func(out, R)
                if losses is None:
                    losses = torch.zeros(R.shape[0], dtype=R.dtype, device=R.device)
                logger_eval.add_samples(image, losses, out.view(-1,3,3), R, Rest, class_idx_cpu, hard)
        logger_eval.finish()
        if verbose:
            loggers.save_network(epoch, model)

class TrainSetting():
    def __init__(self, run_name, config):
        self.run_name = run_name
        self.config = config

    def json_serialize(self):
        return {'run_name': self.run_name,
                'config': self.config}

    @staticmethod
    def json_deserialize(dic):
        run_name = dic['run_name']
        config = TrainConfig.json_deserialize(dic['config'])
        return TrainSetting(run_name, config)

class TrainConfig():
    def __init__(self, typ):
        self.type=typ

    @staticmethod
    def json_deserialize(dic):
        if dic['type'] == 'pascal':
            return PascalConfig.json_deserialize(dic)
        elif dic['type'] == 'upna':
            return UPNAConfig.json_deserialize(dic)
        elif dic['type'] == 'modelnet':
            return ModelnetConfig.json_deserialize(dic)
        else:
            raise RuntimeError('Can not deserialize Train config: {}'.format(dic))

    def json_serialize(self):
        raise RuntimeError('can not serialize abstract class')

class PascalConfig(TrainConfig):
    # data_aug is bool
    # embedding_dim is int
    # synthetic_data is bool
    # warp is bool
    def __init__(self, data_aug, embedding_dim, synthetic_data, warp, pascal_train):
        super().__init__('pascal')
        self.data_aug = data_aug
        self.embedding_dim = embedding_dim
        self.synthetic_data = synthetic_data
        self.warp = warp
        self.pascal_train = pascal_train

    @staticmethod
    def json_deserialize(dic):
        data_aug = dic['data_aug']
        embedding_dim = dic['embedding_dim']
        synthetic_data = dic['synthetic_data']
        warp = dic['warp']
        pascal_train = dic['pascal_train']
        return PascalConfig(data_aug, embedding_dim, synthetic_data, warp, pascal_train)

    def json_serialize(self):
        return {'type': 'pascal',
                'data_aug': self.data_aug,
                'embedding_dim': self.embedding_dim,
                'synthetic_data': self.synthetic_data,
                'warp': self.warp,
                'pascal_train': self.pascal_train}

class ModelnetConfig(TrainConfig):
    # embedding_dim is int
    def __init__(self, embedding_dim):
        super().__init__('modelnet')
        self.embedding_dim = embedding_dim

    @staticmethod
    def json_deserialize(dic):
        return ModelnetConfig(dic['embedding_dim'])

    def json_serialize(self):
        return {'type': 'modelnet',
                'embedding_dim': self.embedding_dim}

class UPNAConfig(TrainConfig):
    def __init__(self):
        super().__init__('upna')
        self.embedding_dim = 0

    @staticmethod
    def json_deserialize(dic):
        return UPNAConfig()

    def json_serialize(self):
        return {'type': 'upna'}

def parse_config():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('run_name', type=str, default='dummy')
    arg_parser.add_argument('config_file', type=str)
    args = arg_parser.parse_args()
    run_name = args.run_name
    config_file = args.config_file
    with open(config_file, 'rb') as f:
        config_dict = json.load(f)
    config = TrainConfig.json_deserialize(config_dict)
    training_setting = TrainSetting(run_name, config)
    return training_setting


import shutil
def main():
    train_setting = parse_config()
    train_model(vmf_loss, 9, train_setting)

if __name__ == '__main__':
    main()
