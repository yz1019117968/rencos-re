#!/usr/bin/env python
#-*- coding:utf-8 -*-
# file name: semantic.py
# file description:
# created at:13/10/2021 4:23 PM

"""
Usage:
  semantic.py [options] MODEL_PATH VOCAB_FILE TRAIN_SET_SRC TRAIN_SET_TGT TEST_SET_SRC TEST_SET_TGT

Options:
    -h --help                 show this screen.
    --cuda INT                use GPU [default: true]
    --embed-size INT          embed size [default: 256]
    --enc-hidden-size INT     encoder hidden size [default: 512]
    --num-layers INT          number of layers [default: 1]
    --dropout-rate FLOAT      dropout rate [default: 0.2]
    --src-max-len INT         max length of src [default: 100]
    --tgt-max-len INT         max length of tgt [default: 50]
"""

from modules.Encoder import Encoder
from docopt import docopt
from vocab import Vocab
from typing import Tuple
from torch import nn
from modules.utils.common import *
from train import Procedure
from dataset import Dataset

class Retriever(Procedure):

    def __init__(self, args):
        super(Retriever, self).__init__(args)
        self._args = args
        self._init_model()

    def _init_model(self):
        vocab = Vocab.load(self._args['VOCAB_FILE'])
        src_vocab = vocab.src_vocab
        enc_embed_layer = nn.Embedding(len(src_vocab), int(self._args['--embed-size']), padding_idx=src_vocab[PADDING])
        self._model = Encoder(int(self._args['--embed-size']), int(self._args['--enc-hidden-size']), enc_embed_layer,
                              int(self._args['--num-layers']), float(self._args['--dropout-rate']))
        trained_params = torch.load(self._args['MODEL_PATH'], map_location=lambda storage, loc: storage)
        # select trained params
        new_params = {k[8:]: v for k, v in trained_params['state_dict'].items() if k.startswith('encoder')}
        # init params of encoder
        init_params = self._model.state_dict()
        # update params of encoder
        init_params.update(new_params)
        # load trained params for encoder
        self._model.load_state_dict(new_params)

    def retrieve(self):
        """
        retrieve the most similar code based on code semantics, then save the source code and corresponding summaries.
        :param args:
        :return:
        """
        train_set, test_set = self.prepare_dataset()
    #     todo
    def prepare_dataset(self):
        train_set = Dataset.create_from_file(self._args['TRAIN_SET_SRC'], self._args['TRAIN_SET_TGT'],
                                             self._args['--src-max-len'], self._args['--tgt-max-len'])
        test_set = Dataset.create_from_file(self._args['TEST_SET_SRC'], self._args['TEST_SET_TGT'],
                                            self._args['--src-max-len'], self._args['--tgt-max-len'])
        return train_set, test_set

    def compare(self):
        pass

def main():
    args = docopt(__doc__)
    print(args)
    retriever = Retriever(args)


if __name__ == "__main__":
    main()
