# import torch
import numpy as np

class MyDataSet(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

class MyDataLoader:
    def __init__(self, data_set, batch_size=1, shuffle=False, drop_last=False):
        self.data = data_set
        self.num = len(data_set)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.max_steps = None
        self.step = None
        self.shf_index = None
    
    def __iter__(self):
        self.max_steps=self.__len__()
        self.step=0
        self.shf_index = np.arange(self.num)
        if self.shuffle:
            self.shf_index = np.random.permutation(self.shf_index)
            
        return self

    def __next__(self):
        batch_size = self.batch_size
        step=self.step
        if self.step < self.max_steps:
            index=self.shf_index[step * batch_size:
                                 min((step + 1) * batch_size,self.num)]
            data_batch=self.data[index]
            self.step+=1
            return data_batch
        else:
            raise StopIteration
    
    def __len__(self):
        if not self.drop_last or self.num % self.batch_size == 0:
            max_steps = (self.num - 1) // self.batch_size + 1
        else:
            max_steps = self.num // self.batch_size
        return max_steps







