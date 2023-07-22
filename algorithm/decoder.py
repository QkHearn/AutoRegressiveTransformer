#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/17 23:52
# @Author  : Hearn
# @File    : decoder.py
# @Software: PyCharm
import torch
from torch import nn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        ''' No encoder memory information for decoder
        :param d_model:the number of expected features in the input
        :param nhead:the number of heads in the multiheadattention models
        :param dim_feedforward:the dimension of the feedforward network algorithm
        :param dropout:
        '''
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Position-wise Feedforward Layer
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Layer Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        # norm first
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask)
        tgt = tgt + self.dropout(tgt2)

        # Position-wise Feedforward Sublayer
        tgt2 = self.norm2(tgt)
        tgt2 = self.feedforward(tgt2)
        tgt = tgt + self.dropout(tgt2)
        return tgt


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, dim_feedforward, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, dim_feedforward, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, tgt_mask):
        for layer in self.layers:
            tgt = layer(tgt, tgt_mask=tgt_mask)
        tgt = self.norm(tgt)
        return tgt
