import numpy as np
import utils

class Trainset():
    def __init__(self):
        self.data = None
        self.labels = None
        self.current_index = 0

    def next_batch(self,size):
        assert size < len(self.data), 'train set is too small!'
        batch = self.data[self.current_index:self.current_index+size]
        self.current_index += size
        self.current_index = self.current_index % len(self.data)
        return batch

class Testset():
    def __init__(self):
        self.data = None
        self.labels = None

class Dataset():
    def __init__(self, k):
        self.k = k
        self.train = Trainset()
        self.test = Testset()

    def setTrainData(self,data, labels = None):
        self.train.data = data
        if type(labels) is np.ndarray:
            self.train.labels = utils.one_hot(labels, self.k)

    def setTestData(self, data, labels = None):
        self.test.data = data
        if type(labels) is np.ndarray:
            self.test.labels = utils.one_hot(labels, self.k)
