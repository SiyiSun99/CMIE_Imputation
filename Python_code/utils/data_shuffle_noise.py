#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   data shuffle and noise
@Author      :   siyi.sun
@Time        :   2025/02/21 01:01:04
"""

import numpy as np


class data_shuffle_noise:
    def __init__(self, mode, noise_zero, high):
        self.mode = mode
        self.noise_zero = noise_zero
        self.high = high
        self.low = 0.0

    def _add_noise_(self, train_data, train_mask):
        if self.noise_zero:
            train_data[np.isnan(train_data)] = 0.0
            return train_data
        else:
            noise = np.random.uniform(
                low=self.low, high=self.high, size=train_data.shape
            )
            train_data[np.isnan(train_data)] = 0.0
            return train_data * train_mask + noise * (1.0 - train_mask)

    def dataNonenan(self, data, mask):
        index = []
        for i in range(data.shape[0]):
            if str(np.sum(data[i, :])) != "nan":
                index.append(i)
        print(data[index, :].shape[0])
        return data[index, :], mask[index, :]

    def return_noise_batch(self, shape):
        if self.noise_zero:
            return np.zeros(shape, dtype=np.float32)
        else:
            return np.random.uniform(low=-self.high, high=self.high, size=shape)

    def data_shuffle(self, train_data, train_mask):
        idx = np.arange(train_data.shape[0])
        np.random.shuffle(idx)
        train_d = train_data[idx]
        train_m = train_mask[idx]
        train_d = self._add_noise_(train_data=train_d, train_mask=train_m)
        return train_d, train_m
