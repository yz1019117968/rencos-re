#!/usr/bin/env python
#-*- coding:utf-8 -*-
# file name: Encoder.py
# file description:
# created at:29/9/2021 2:17 PM

from abc import ABC, abstractmethod
from typing import List, Tuple
import torch
from torch import nn, Tensor
from modules.base import LSTM


class BaseEncoder(nn.Module, ABC):
    def __init__(self):
        super(BaseEncoder, self).__init__()
        self.embed_layer = None

    @property
    @abstractmethod
    def output_size(self):
        pass


class Encoder(BaseEncoder):
    def __init__(self, embed_size, hidden_size, embed_layer, num_layers, dropout_rate):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embed_layer = embed_layer
        self.rnn_layer = LSTM(self.embed_size, self.hidden_size, num_layers, bidirectional=True, batch_first=False,
                              dropout=dropout_rate)

    @property
    def output_size(self):
        """
        :return: hidden_size*num_directions
        """
        return self.hidden_size * 2

    def forward(self, src_tensor: torch.Tensor, src_lens: List[int]):
        """
        :param src_tensor: (src_sent_len， batch_size)
        :param src_lens:
        :return: (src_sent_len, batch_size, hidden_size*num_directions),
                (num_layers*num_directions, batch_size, hidden_size),
                 (num_layers*num_directions, batch_size, hidden_size)
        """
        # (batch_size, sent_len, embed_size) -> (sent_len, batch_size, embed_size)
        # tell diff about view and permute

        embeddings = self.embed_layer(src_tensor.permute(1, 0))
        embeddings = embeddings.permute(1, 0, 2)
        encodings, (last_state, last_cell) = self.rnn_layer(embeddings, src_lens, enforce_sorted=False)

        encodings = encodings.view(encodings.size(0), encodings.size(1), 2, self.hidden_size)
        encodings = encodings.sum(2).permute(1, 0, 2)
        last_state = last_state.sum(0).unsqueeze(0)
        last_cell = last_cell.sum(0).unsqueeze(0)
        return encodings, last_state, last_cell

if __name__ == "__main__":
    embed_layer = nn.Embedding(10, 3)
    encoder = Encoder(3, 5, embed_layer, 1, 0.5)
    src_tensor = torch.tensor([[1,2,3,4], [2,3,4,0]])
    src_lens = [4, 3]
    print(encoder(src_tensor, src_lens))
