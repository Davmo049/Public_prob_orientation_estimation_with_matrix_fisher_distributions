from Pascal3D import Pascal3D_render, Pascal3D
from torch.utils.data import Dataset, IterableDataset, Subset, ConcatDataset

import numpy as np

class RandomSubsetDataset(Dataset):
    def __init__(self, dataset, num_samples):
        assert(len(dataset) >= num_samples)
        self.num_samples = num_samples
        self.dataset = dataset

    def __getitem__(self, idx):
        idx = np.random.randint(len(self.dataset))
        # intentional override of idx, sample uniformly with replacement instead
        # One could create a new random subset per epoch instead, but then there might be threading issues
        # This is fine
        return self.dataset[idx]

    def __len__(self):
        return self.num_samples


class Pascal3DAll():
    def __init__(self, real, rendered, rendered_percentage=0.2):
        rendered_subset = RandomSubsetDataset(rendered, int(len(rendered)*rendered_percentage))
        print(len(rendered))
        self.train_dataset = ConcatDataset([real.get_train(augmentation=True), rendered_subset])
        self.real_ds = real

    def get_train(self):
        return self.train_dataset

    def get_real_train(self, augmentation=True):
        return self.real_ds.get_train(augmentation=augmentation)

    def get_eval(self):
        return self.real_ds.get_eval()
