#!/usr/bin/env python
#-*- coding:utf-8 -*-
# file name: base.py
# file description:
# created at:30/9/2021 11:50 AM

from typing import List, Tuple

import torch
import logging
from abc import ABC, abstractmethod
from torch import nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

logging.basicConfig(level=logging.INFO)


class BaseModel(nn.Module, ABC):
    def __init__(self, *args):
        super(BaseModel, self).__init__()

    @staticmethod
    @abstractmethod
    def prepare_model_params(args: dict):
        """
        construct parameters for __init__ function from model args
        """
        pass

    @classmethod
    def load(cls, model_path: str):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        print("TYPE: ", type(params))
        args = params['args']
        model = cls(*args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str, args: dict):
        logging.info('save model parameters to [%s]' % path)
        model_args = self.prepare_model_params(args)
        logging.info('model args:\n{}'.format(model_args))
        params = {
            'args': model_args,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bidirectional: bool, batch_first: bool,
                 dropout: float):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, x_lens: List[int], init_states: Tuple[Tensor, Tensor] = None, enforce_sorted: bool = True) \
            -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        packed_input = pack_padded_sequence(x, x_lens, batch_first=self.rnn.batch_first, enforce_sorted=enforce_sorted)
        encodings, (last_state, last_cell) = self.rnn(packed_input)
        encodings, _ = pad_packed_sequence(encodings, batch_first=self.rnn.batch_first)
        return encodings, (last_state, last_cell)




class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float, bias: bool = True):
        super(LSTMCell, self).__init__()
        self.rnn_cell = nn.LSTMCell(input_size=input_size,
                                    hidden_size=hidden_size,
                                    bias=bias)
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, h_tm1: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        h_t, cell_t = self.rnn_cell(x, h_tm1)
        return h_t, cell_t


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0, bias: bool = True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return self.linear(x)

if __name__ == "__main__":
    lstm = LSTM(3, 4, 1, bidirectional=True, batch_first=False, dropout=0.5)
    x = torch.tensor([[[1,1,1],[2,2,2]],
                      [[3,3,3],[0,0,0]],
                      [[5,5,5],[0,0,0]]]
                     ).permute(1,0,2)
    print(x)
    lstm(x, [2,1,1], enforce_sorted=True)


