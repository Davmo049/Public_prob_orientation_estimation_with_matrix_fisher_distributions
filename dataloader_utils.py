import torch
import numpy as np

def get_concatenated_dataset(dataset_sampler_pairs):
    # dataset_sampler_pairs is a list of (dataset, sampler)
    sampler = RandomConcatenatedSampler(dataset_sampler_pairs)
    return sampler.dataset, sampler

class RandomConcatenatedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset_sampler_pairs):
        # dataset_sampler_pairs is a list of (dataset, sampler)
        dataset_list = list(map(lambda t: t[0], dataset_sampler_pairs))
        dataset = torch.utils.data.ConcatDataset(dataset_list)
        samplers = list(map(lambda t: t[1], dataset_sampler_pairs))
        sampler_lengths = list(map(len, samplers))
        sampler_lengths = [0] + sampler_lengths
        self.samplers = samplers
        self.offsets = np.cumsum(sampler_lengths)
        self.dataset = dataset

    def __iter__(self):
        return RandomConcatenatedSamplerIterator(self)

    def __len__(self):
        return self.offsets[-1]

class RandomConcatenatedSamplerIterator():
    def __init__(self, origin):
        self.origin = origin
        arr = np.empty(len(origin), dtype=np.int)
        for i in range(len(self.origin.offsets)-1):
            offset_start = self.origin.offsets[i]
            offset_end = self.origin.offsets[i+1]
            arr[offset_start:offset_end] = i
        np.random.shuffle(arr)
        self.dataset_to_use = arr
        self.idx = 0
        self.iters = list(map(iter, self.origin.samplers))

    def __next__(self):
        if self.idx >= len(self.dataset_to_use):
            raise StopIteration()
        ds_idx = self.dataset_to_use[self.idx]
        offset_from_start = next(self.iters[ds_idx])
        self.idx += 1
        concat_idx = offset_from_start + self.origin.offsets[ds_idx]
        return concat_idx

class RandomSubsetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, num_samples):
        self.tot_samples = len(dataset)
        self.num_samples = num_samples
        assert(self.tot_samples >= self.num_samples)
        assert(self.num_samples > 0)

    def __iter__(self):
        all_samples = np.arange(self.tot_samples)
        np.random.shuffle(all_samples)
        return iter(all_samples[:self.num_samples])

    def __len__(self):
        return self.num_samples
