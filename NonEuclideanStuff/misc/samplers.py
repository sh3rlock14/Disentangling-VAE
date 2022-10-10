from torch.utils.data.sampler import Sampler
from numpy.random import randint
from random import shuffle

class IntraSampler(Sampler):
    def __init__(self, dataset):
        self.class_ids = dataset.class_ids
    
    def __iter__(self):
        class_idx = self.class_ids[randint(len(self.class_ids))]
        class_idxs = [i for i, id in enumerate(self.class_ids) if id==class_idx]
        shuffle(class_idxs)
        return iter(class_idxs)
    
    def __len__(self):
        print('\tcalling IntraSampler:__len__')
        return self.num_samples
    