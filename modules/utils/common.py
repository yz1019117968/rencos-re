#!/usr/bin/env python
#-*- coding:utf-8 -*-
# file name: common.py
# file description:
# created at:30/9/2021 1:03 PM


import re
import torch
import random
import importlib
import numpy as np
from typing import List, Iterable


PADDING = '<pad>'
CODE_PAD = '<pad>'
TGT_START = '<s>'
TGT_END = '</s>'
UNK = '<unk>'

FLOAT_TYPE = torch.float


def set_reproducibility(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    # set random seed for all devices (both CPU and GPU)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ids_to_input_tensor(word_ids: List[List[int]], pad_token: int, device: torch.device) -> torch.Tensor:
    sents_t = input_transpose(word_ids, pad_token)
    sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
    return sents_var


def input_transpose(sents, pad_token):
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t


def get_attr_by_name(class_name: str):
    class_tokens = class_name.split('.')
    assert len(class_tokens) > 1
    module_name = ".".join(class_tokens[:-1])
    module = importlib.import_module(module_name)
    print("module: ", module)
    return getattr(module, class_tokens[-1])
