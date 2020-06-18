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
matplotlib.use('Agg')

def vmf_loss(net_out, R, overreg=1.05):
    A = net_out.view(-1,3,3)
    loss_v = loss.KL_approx(A, R, overreg=overreg)
    if loss_v is None:
        Rest = torch.unsqueeze(torch.eye(3,3, device=R.device, dtype=R.dtype), 0)
        Rest = torch.repeat_interleave(Rest, R.shape[0], 0)
        return None, Rest

    Rest = loss.batch_torch_A_to_R(A)
    return loss_v, Rest


def training_modelnetso3(loss_func, out_dim, run_index):
    # device = 'cpu'
    device = 'cuda'
    batch_size = 32
    base = resnet101(pretrained=True, progress=True)
    model = ResnetHead(base, 10, 32, 512, out_dim)
    model.to(device)
    dataset = ModelNetSo3.ModelNetSo3('/local_storage/datasets')
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
        drop_last=False,
        num_workers=0, # I suspect the suprocess fails to free their transactions when terminating? not too much processing done in dataloader anyways
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True)

    base_lr = 0.01
    if model.class_embedding is None:
        finetune_parameters = model.head.parameters()
    else:
        finetune_parameters = list(model.head.parameters()) + list(model.class_embedding.parameters())
    opt = torch.optim.SGD(finetune_parameters, lr=base_lr)
    loggers = logger.Logger('logs/Modelnet/{}'.format(run_index), ModelNetSo3.ModelNetSo3Classes, 30, dataset)
    num_epochs = 50
    for epoch in range(num_epochs):
        verbose = epoch % 10 == 0 or epoch == num_epochs - 1
        if epoch == 45:
            opt = torch.optim.SGD(model.parameters(), lr=0.001*base_lr)
        elif epoch == 40:
            opt = torch.optim.SGD(model.parameters(), lr=0.01*base_lr)
        elif epoch == 30:
            opt = torch.optim.SGD(model.parameters(), lr=0.1*base_lr)
        elif epoch == 2:
            opt = torch.optim.SGD(model.parameters(), lr=base_lr)

        train_logger = loggers.get_train_logger(epoch, verbose)
        model.train()
        for image, extrinsic, class_idx_cpu, hard, _, _ in tqdm.tqdm(dataloader_train):
            image = image.to(device)
            R = extrinsic[:, :3,:3].to(device)
            class_idx = class_idx_cpu.to(device)
            out = model(image, class_idx)
            losses, Rest = loss_func(out, R)
            if losses is not None:
                loss = torch.mean(losses)
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                losses = torch.zeros(R.shape[0], dtype=R.dtype, device=device)
            train_logger.add_samples(image, losses, out.view(-1,3,3), R, Rest, class_idx_cpu, hard)
        train_logger.finish()
        train_logger = None

        image=None
        R=None
        class_idx = None
        out = None
        losses=None
        loss=None
        Rest=None

        eval_logger = loggers.get_validation_logger(epoch, verbose)
        model.eval()
        with torch.no_grad():
            for image, extrinsic, class_idx_cpu, hard, _, _ in tqdm.tqdm(dataloader_eval):
                image = image.to(device)
                R = extrinsic[:,:3,:3].to(device)
                class_idx = class_idx_cpu.to(device)
                out = model(image, class_idx)
                losses, Rest = loss_func(out, R)
                if losses is None:
                    print('losses none in eval!')
                    losses = torch.zeros(R.shape[0], dtype=R.dtype, device=R.device)
                eval_logger.add_samples(image, losses, out.view(-1,3,3), R, Rest, class_idx_cpu, hard)
        eval_logger.finish()

        if verbose:
            loggers.save_network(epoch, model)




def training_upna(loss_func, out_dim, run_index):
    # device = 'cpu'
    device = 'cuda'
    batch_size = 32
    base = resnet101(pretrained=True, progress=True)
    model = ResnetHead(base, 1, 0, 512, out_dim)
    model.to(device)
    dataset = UPNA.UPNA('/local_storage/datasets')
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
        pin_memory=True)

    base_lr = 0.01
    finetune_param = model.head.parameters()
    finetune_parameters = model.head.parameters()
    opt = torch.optim.SGD(finetune_parameters, lr=base_lr)
    loggers = logger.Logger('logs/UPNA/{}'.format(run_index), Pascal3D.PascalClasses, 30, dataset)
    num_epochs = 120
    for epoch in range(num_epochs):
        verbose = epoch % 10 == 0 or epoch == num_epochs-1
        if epoch == 90:
            opt = torch.optim.SGD(model.parameters(), lr=0.001*base_lr)
        elif epoch == 60:
            opt = torch.optim.SGD(model.parameters(), lr=0.01*base_lr)
        elif epoch == 30:
            opt = torch.optim.SGD(model.parameters(), lr=0.1*base_lr)
        elif epoch == 3:
            opt = torch.optim.SGD(model.parameters(), lr=base_lr)

        logger_train = loggers.get_train_logger(epoch, verbose)
        model.train()
        for image, extrinsic, class_idx_cpu, hard, _, _ in dataloader_train:
            image = image.to(device)
            R = extrinsic[:, :3,:3].to(device)
            class_idx = class_idx_cpu.to(device)
            out = model(image, class_idx-1)
            losses, Rest = loss_func(out, R)

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
                out = model(image, class_idx-1)
                losses, Rest = loss_func(out, R)
                if losses is None:
                    losses = torch.zeros(R.shape[0], dtype=R.dtype, device=R.device)
                logger_eval.add_samples(image, losses, out.view(-1,3,3), R, Rest, class_idx_cpu, hard)
        logger_eval.finish()
        if verbose:
            loggers.save_network(epoch, model)



def training_pascal(loss_func, out_dim, run_index):
    # device = 'cpu'
    device = 'cuda'
    batch_size = 32
    base = resnet101(pretrained=True, progress=True)
    model = ResnetHead(base, 12, 0, 512, out_dim)
    model.to(device)
    use_synthetic_data = False
    use_augmentation = True
    use_warp = True
    if not use_warp:
        assert(not use_synthetic_data)
        assert(use_augmentation is None)
    dataset_real = Pascal3D.Pascal3D('/local_storage/datasets', train_all=True, use_warp=use_warp) # switch back when making decisions
    if use_synthetic_data:
        train_real = dataset_real.get_train(use_augmentation)
        real_sampler = torch.utils.data.sampler.RandomSampler(train_real, replacement=False)
        dataset_rendered = Pascal3D_render.Pascal3DRendered('/local_storage/datasets')
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
    else:
        dataloader_train = torch.utils.data.DataLoader(
            dataset_real.get_train(use_augmentation),
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
            pin_memory=True,
            drop_last=True)

    dataset = dataset_real
    dataloader_eval = torch.utils.data.DataLoader(
        dataset_real.get_eval(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True)

    base_lr = 0.01
    if model.class_embedding is None:
        finetune_parameters = model.head.parameters()
    else:
        finetune_parameters = list(model.head.parameters()) + list(model.class_embedding.parameters())

    opt = torch.optim.SGD(finetune_parameters, lr=base_lr)
    loggers = logger.Logger('logs/Pascal/{}'.format(run_index), Pascal3D.PascalClasses, 5, dataset)
    num_epochs = 120
    for epoch in range(num_epochs):
        verbose = epoch % 10 == 0 or epoch == num_epochs-1
        if epoch == 90:
            opt = torch.optim.SGD(model.parameters(), lr=0.001*base_lr)
        elif epoch == 60:
            opt = torch.optim.SGD(model.parameters(), lr=0.01*base_lr)
        elif epoch == 30:
            opt = torch.optim.SGD(model.parameters(), lr=0.1*base_lr)
        elif epoch == 3:
            opt = torch.optim.SGD(model.parameters(), lr=base_lr)

        logger_train = loggers.get_train_logger(epoch, verbose)
        model.train()
        for image, extrinsic, class_idx_cpu, hard, _, _ in dataloader_train:
            image = image.to(device)
            R = extrinsic[:, :3,:3].to(device)
            class_idx = class_idx_cpu.to(device)
            out = model(image, class_idx-1)
            losses, Rest = loss_func(out, R)

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
                out = model(image, class_idx-1)
                losses, Rest = loss_func(out, R)
                if losses is None:
                    losses = torch.zeros(R.shape[0], dtype=R.dtype, device=R.device)
                logger_eval.add_samples(image, losses, out.view(-1,3,3), R, Rest, class_idx_cpu, hard)
        logger_eval.finish()
        if verbose:
            loggers.save_network(epoch, model)

import shutil
def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('run_index', type=str, default='dummy')
    args = arg_parser.parse_args()
    run_index = args.run_index
    training_pascal(vmf_loss, 9, run_index)
    training_upna(vmf_loss, 9, run_index)
    training_modelnetso3(vmf_loss, 9, run_index)

if __name__ == '__main__':
    main()
