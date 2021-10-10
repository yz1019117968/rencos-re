#!/usr/bin/env python
#-*- coding:utf-8 -*-
# file name: dataset.py
# file description:
# created at:30/9/2021 1:44 PM

import math
from abc import ABC, abstractmethod
from torch import Tensor
from typing import Callable
from modules.utils.common import *
import logging
from vocab import VocabEntry

logging.basicConfig(level=logging.INFO)


class AbstractExample(ABC):
    @property
    @abstractmethod
    def src_tokens(self):
        pass

    @property
    @abstractmethod
    def tgt_tokens(self):
        pass


class Example(AbstractExample):
    def __init__(self, instance):
        self._sample_id = instance['sample_id']
        self._src_tokens = instance['src_tokens']
        self._tgt_tokens = instance['tgt_tokens']

    # 只读属性
    @property
    def src_tokens(self):
        """
        used for models
        """
        return self._src_tokens

    @property
    def tgt_tokens(self):
        """
        for creating vocab
        """
        return [TGT_START] + self._tgt_tokens + [TGT_END]

    @property
    def tgt_in_tokens(self):
        return [TGT_START] + self._tgt_tokens

    @property
    def tgt_out_tokens(self):
        return self._tgt_tokens + [TGT_END]

    @property
    def get_tgt_desc_tokens(self):
        return self._tgt_tokens

    # for validate step, the predicted word number is len(self.tgt_tokens) - 1
    @property
    def tgt_words_num(self):
        return len(self.tgt_tokens) - 1

# class TestExample(AbstractExample):
#     def __init__(self, instance):
#         self._sample_id = instance['sample_id']
#         self._src_tokens = instance['src_tokens']
#         self._src_tokens_0 = instance['src_tokens_0']
#         self._src_tokens_1 = instance['src_tokens_1']
#         self._tgt_tokens = instance['tgt_tokens']
#
#     # 只读属性
#     @property
#     def src_tokens(self):
#         """
#         used for models
#         """
#         return self._src_tokens
#
#     @property
#     def src_tokens_0(self):
#         """
#         used for models
#         """
#         return self._src_tokens_0
#
#     @property
#     def src_tokens_1(self):
#         """
#         used for models
#         """
#         return self._src_tokens_1
#
#     @property
#     def tgt_tokens(self):
#         """
#         used for models
#         """
#         return [TGT_START] + self._tgt_tokens + [TGT_END]

class Batch(object):

    def __init__(self, examples: List[Example]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item) -> Example:
        return self.examples[item]

    # for validate step
    @property
    def tgt_words_num(self) -> int:
        return sum([e.tgt_words_num for e in self.examples])

    @property
    def src_tokens(self):
        return [e.src_tokens for e in self.examples]

    def get_src_lens(self):
        return [len(sent) for sent in self.src_tokens]

    @property
    def tgt_tokens(self):
        return [e.tgt_tokens for e in self.examples]

    def get_tgt_lens(self):
        return [len(sent) for sent in self.tgt_tokens]

    @property
    def tgt_in_tokens(self):
        return [e.tgt_in_tokens for e in self.examples]

    def get_tgt_in_lens(self):
        return [len(sent) for sent in self.tgt_in_tokens]

    @property
    def tgt_out_tokens(self):
        return [e.tgt_out_tokens for e in self.examples]

    def get_tgt_out_lens(self):
        return [len(sent) for sent in self.tgt_out_tokens]

    def get_src_tensor(self, vocab: VocabEntry, device: torch.device):
        return vocab.to_input_tensor(self.src_tokens, device)

    def get_tgt_tensor(self, vocab: VocabEntry, device: torch.device) -> Tensor:
        return vocab.to_input_tensor(self.tgt_tokens, device)

    def get_tgt_in_tensor(self, vocab: VocabEntry, device: torch.device):
        return vocab.to_input_tensor(self.tgt_in_tokens, device)

    def get_tgt_out_tensor(self, vocab: VocabEntry, device: torch.device):
        return vocab.to_input_tensor(self.tgt_out_tokens, device)

class Dataset(object):
    def __init__(self, examples: List[Example]):
        self.examples = examples

    @staticmethod
    def create_from_file(src_file_path: str, tgt_file_path: str, src_max_len, tgt_max_len, ExampleClass: Callable = Example):
        """
        construct the dataset.
        """
        import os
        print("abs path: ", os.path.split(os.path.realpath(__file__))[0])
        src_entries = []
        with open(src_file_path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                src_tokens = line.strip().split()
                if src_max_len is not None and int(src_max_len) < len(src_tokens):
                    src_tokens = line.strip().split()[: int(src_max_len)]
                src_entries.append(src_tokens)
        tgt_entries = []
        with open(tgt_file_path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                tgt_tokens = line.strip().split()
                if tgt_max_len is not None and int(tgt_max_len) < len(tgt_tokens):
                    tgt_tokens = line.strip().split()[: int(tgt_max_len)]
                tgt_entries.append(tgt_tokens)
        examples = []
        for idx, (src, tgt) in enumerate(zip(src_entries, tgt_entries)):
            examples.append(ExampleClass({"sample_id": idx, "src_tokens": src, "tgt_tokens": tgt}))
        logging.info("loading {} samples".format(len(examples)))
        return Dataset(examples)

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.examples)

    def get_src_tokens(self):
        for e in self.examples:
            yield e.src_tokens

    def get_tgt_tokens(self):
        for e in self.examples:
            yield e.tgt_tokens

    def get_ground_truth(self) -> Iterable[List[str]]:
        for e in self.examples:
            # remove the <s> and </s>
            yield e.get_tgt_desc_tokens

    def _batch_iter(self, batch_size: int, shuffle: bool, sort_by_length: bool) -> Batch:
        batch_num = math.ceil(len(self) / batch_size)
        index_array = list(range(len(self)))

        if shuffle:
            np.random.shuffle(index_array)

        for i in range(batch_num):
            indices = index_array[i * batch_size: (i + 1) * batch_size]
            examples = [self[idx] for idx in indices]

            if sort_by_length:
                examples = sorted(examples, key=lambda e: len(e.src_tokens), reverse=True)
            yield Batch(examples)

    def train_batch_iter(self, batch_size: int, shuffle: bool) -> Batch:
        for batch in self._batch_iter(batch_size, shuffle=shuffle, sort_by_length=True):
            yield batch

    def infer_batch_iter(self, batch_size):
        for batch in self._batch_iter(batch_size, shuffle=False, sort_by_length=False):
            yield batch

if __name__ == "__main__":
    instance = {"sample_id": 0, "src_tokens": [1,2,4,56,7], "tgt_tokens": [3,4,7,8,4]}


