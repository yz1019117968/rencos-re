#!/usr/bin/env python
#-*- coding:utf-8 -*-
# file name: vocab.py
# file description:
# created at:5/10/2021 10:26 AM

"""
Usage:
    vocab.py --train-set-src=<file> --train-set-tgt=<file> [options] VOCAB_FILE
Options:
    -h --help                  Show this screen.
    --train-set-src=<file>     Train set file (src)
    --train-set-tgt=<file>     Train set file (tgt)
    --size-src=<int>           src vocab size [default: 50000]
    --size-tgt=<int>           tgt vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
    --vocab-class=<str>        the class name of used Vocab class [default: Vocab]
"""

import os
import pickle
from abc import ABC, abstractmethod
from typing import Iterable, TYPE_CHECKING, Union
from collections import Counter
from itertools import chain
from docopt import docopt
import json
from utils.common import *
from tqdm import tqdm
if TYPE_CHECKING:
    from dataset import Dataset


class BaseVocabEntry(ABC):
    def __init__(self):
        self.word2id = None
        self.id2word = None

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def index2word(self, wid):
        pass

    @abstractmethod
    def indices2words(self, word_ids: List[int]):
        pass

    @abstractmethod
    def words2indices(self, sents: Union[List[str], List[List[str]]]):
        pass


class VocabEntry(BaseVocabEntry):
    def __init__(self, word2id=None):
        super(VocabEntry, self).__init__()
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id[PADDING] = 0
            self.word2id[TGT_START] = 1
            self.word2id[TGT_END] = 2
            self.word2id[UNK] = 3

        self.unk_id = self.word2id[UNK]

        self.id2word = {v: k for k, v in self.word2id.items()}
        self.embeddings = None

    def __getitem__(self, word):
        """
        return values with "word", if if none, return self.unk_id, value = obj[word]?obj[word]:self.unk_id
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """
        return whether the "word" is in the obj by opt "in", word in obj
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """
        write values into the obj with "key" and "value", obj[key] = value
        """
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        """
        return the length, len(obj)
        """
        return len(self.word2id)

    def __repr__(self):
        """
        print(obj)
        """
        return 'Vocabulary[size=%d]' % len(self)

    def index2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        """
        add new word into vocab, return word id
        :param word:
        :return:
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """
        :param sents: List or List[List]
        :return:
        """
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids) -> List[str]:
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        word_ids = self.words2indices(sents)
        sents_var = ids_to_input_tensor(word_ids, self[PADDING], device)
        return sents_var

    @staticmethod
    def from_corpus(corpus: Iterable[List[str]], size: int, freq_cutoff=2):
        """
        construct vocab from corpus
        """
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print(
            f'number of word types: {len(word_freq)}, number of word types w/ frequency >= {freq_cutoff}: {len(valid_words)}')

        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry

class Vocab(object):
    def __init__(self, src_vocab: VocabEntry, tgt_vocab: VocabEntry):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    @staticmethod
    def build(dataset: 'Dataset', src_vocab_size: int, tgt_vocab_size: int, src_freq_cutoff: int, tgt_freq_cutoff: int) -> 'Vocab':
        print('initialize src vocabulary..')
        src_vocab = VocabEntry.from_corpus(dataset.get_src_tokens(), src_vocab_size, src_freq_cutoff)
        print('initialize tgt vocabulary..')
        tgt_vocab = VocabEntry.from_corpus(dataset.get_tgt_tokens(), tgt_vocab_size, tgt_freq_cutoff)
        return Vocab(src_vocab, tgt_vocab)

    def save(self, file_path: str):
        assert file_path.endswith(".json")
        with open(file_path, 'w') as f:
            json.dump(dict(code_word2id=self.src_vocab.word2id,
                           nl_word2id=self.tgt_vocab.word2id), f, indent=2)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as f:
            entry = json.load(f)
        src_word2id = entry['src_word2id']
        tgt_word2id = entry['tgt_word2id']
        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

    def __repr__(self):
        return 'Vocab(src %d words, tgt %d words)' % (len(self.src_vocab), len(self.tgt_vocab))

if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)
    from dataset import Dataset

    print("Loading train set src: " + args['--train-set-src'])
    print("Loading train set tgt: " + args['--train-set-tgt'])
    train_set = Dataset.create_from_file(args['--train-set-src'], args['--train-set-tgt'], None, None)
    print(train_set[0:5])
    # vocab_class = globals()[args['--vocab-class']]
    # vocab = vocab_class.build(train_set, int(args['--size-src']), int(args['--size-tgt']), int(args['--freq-cutoff']), int(args['--freq-cutoff']))
    # print('generated vocabulary, {}'.format(vocab))
    #
    # vocab.save(args['VOCAB_FILE'])
    # print('vocabulary saved to %s' % args['VOCAB_FILE'])



