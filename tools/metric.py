import numpy as np


class AccuracyMulti(object):
    def __init__(self):
        self.count = 0
        self.correct = 0

    def judge(self, preds, labels):
        self.count += len(labels)
        predicts = np.argmax(preds, axis=1)
        self.correct += np.sum(predicts == labels)

    def summary(self):
        rate = self.correct * 1.0 / self.count
        self.count = 0
        self.correct = 0
        return rate