import numpy as np
import os
import pandas as pd

class TestMatrix():

    def __init__(self, shape, number):
        self.shape = shape
        self.number = number
        self.mat = np.random.randint(0, self.number,(self.shape))

    def __len__(self):
        return

    def getmat(self):
        return self.mat



