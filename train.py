#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/19 23:58
# @Author  : Hearn
# @File    : train.py
# @Software: PyCharm
import torch
from torch import nn
from torch.utils.data import DataLoader

from algorithm.decoder import Decoder
from dataloader import FinanceDataset

EPOCHES = 100
BATCH_SIZE = 10

train_data = FinanceDataset(30, 7, start="2000-01-01", end="2017-07-01")
val_data = FinanceDataset(30, 7, start="2017-01-01", end="2021-07-01")


def generate_square_subsequent_mask(tgt, device):
    return torch.tril(torch.ones((tgt.size(0), tgt.size(0)), dtype=torch.uint8)) \
        .unsqueeze(0).repeat(tgt.size(1), 1, 1).float().to(device)


def train(model, criterion, optimizer):
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    for epoch in range(EPOCHES):
        model.train()
        train_loss = 0
        for idx, (data, label) in enumerate(train_loader):
            data = data.permute(1, 0).unsqueeze(2).to(device)
            label = label.permute(1, 0).unsqueeze(2).to(device)
            mask = generate_square_subsequent_mask(data, device)
            optimizer.zero_grad()
            output = model(data, mask)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if idx % 100 == 0:
                print(f'epoch:{epoch},idx:{idx}, loss:{train_loss}')
        # with torch.no_grad():
        #     for data, label in val_loader:
        #         data = data.to(device)
        #         label = label.to(device)
        #         output = model(data)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = {
        'num_layers': 2,
        'd_model': 1,
        'n_heads': 1,
        'dim_feedforward': 1,
        'dropout': .8,
        'learning_rate': .1
    }
    model = Decoder(model_config['num_layers'],
                    model_config['d_model'],
                    model_config['n_heads'],
                    model_config['dim_feedforward'],
                    model_config['dropout']).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), model_config['learning_rate'])
    train(model, criterion, optimizer)
