import logger
from ModelNetSo3 import ModelNetSo3
import torch
import numpy as np

def main():
    dataset = ModelNetSo3.ModelNetSo3()
    ds_train = dataset.get_train()
    data_loader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True)
    logger_path = 'tmp_logger_dir'
    loggers = logger.Logger(logger_path, ModelNetSo3.ModelNetSo3Classes, len(ds_train)/4, dataset)
    for epoch in range(20):
        logr = loggers.get_train_logger(epoch, epoch % 10 ==0)
        counter = 0
        for image, extrinsic, class_idx_cpu, hard, _, _ in data_loader:
            R = extrinsic[:, :3,:3]
            Rest = torch.zeros(R.shape[0], 3,3)
            Rest[:,0,0]=1
            Rest[:,1,1]=1
            Rest[:,1,1]=1
            prob_params = Rest
            losses = image[:,0,0,0]
            logr.add_samples(image, losses, prob_params, R, Rest, class_idx_cpu, hard)
            if counter == 1:
                break
            counter += 1
        logr.finish()



if __name__ == '__main__':
    main()
