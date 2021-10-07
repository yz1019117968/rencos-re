#!/usr/bin/env python
#-*- coding:utf-8 -*-
# file name: misc.py
# file description:
# created at:4/10/2021 10:08 AM

# -*- coding: utf-8 -*-

import torch
from torch import Tensor

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def use_gpu(opt):
    """
    Creates a boolean if gpu used
    """
    return (hasattr(opt, 'gpu_ranks') and len(opt.gpu_ranks) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)

def negative_log_likelihood(logits: torch.FloatTensor, gold_tensor: torch.LongTensor,
                            words_mask: torch.FloatTensor) -> Tensor:
    """
    :param logits: ( batch_size, tgt_vocab_size), log_softmax
    :param gold_tensor: ([tgt_src_len - 1, x], batch_size)
    :param words_mask: ([tgt_src_len - 1, x], batch_size), a matrix to mask target words, 1.0 for non-pad
                       NOTE: this mask is different from dot-production mask
    :return: losses: ([tgt_src_len - 1, x], batch_size)
    """
    # (sent_len, batch_size)
    gold_words_log_prob = torch.gather(logits, index=gold_tensor.unsqueeze(-1), dim=-1).squeeze(-1) * words_mask
    return -gold_words_log_prob
