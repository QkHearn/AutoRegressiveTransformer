#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/16 11:33
# @Author  : Hearn
# @File    : dataloader.py
# @Software: PyCharm
import numpy as np
import torch
import yfinance as yf
from torch.utils.data import Dataset, DataLoader

# 获取Apple公司的股票数据
data = yf.download("AAPL", start="2010-01-01", end="2021-07-01")


class FinanceDataset(Dataset):
    def __init__(self, sequence_length, windows_size, start="2000-01-01", end="2017-07-01"):
        self.sequence_length = sequence_length
        self.windows_size = windows_size
        self.src, self.trg = self._get_dataset(start, end)

    def __getitem__(self, item):
        return (
            torch.from_numpy(self.src[item]).float(),
            torch.from_numpy(self.trg[item]).float()
        )

    def __len__(self):
        return self.src.shape[0]

    def _get_dataset(self, start, end):
        # return torch
        data = yf.download("AAPL", start=start, end=end)['Volume'].values
        src, trg = [], []
        for i in range(0, len(data) - self.windows_size - self.sequence_length, self.windows_size):
            src.append(data[i:i + self.sequence_length])
            trg.append(data[i + self.windows_size:i + self.windows_size + self.sequence_length])
        return np.array(src), np.array(trg)
