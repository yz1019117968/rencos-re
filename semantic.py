#!/usr/bin/env python
#-*- coding:utf-8 -*-
# file name: semantic.py
# file description:
# created at:13/10/2021 4:23 PM

"""
Usage:
  semantic.py [options] MODEL_PATH VOCAB_FILE TRAIN_SET_SRC TRAIN_SET_TGT TEST_SET_SRC TEST_SET_TGT QUERY_OUT_PATH SOURCE_OUT_PATH

Options:
    -h --help                 show this screen.
    --cuda INT                use GPU [default: False]
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
from dataset import Dataset, Batch
from tqdm import tqdm
import pickle as pkl
import os

class Retriever(Procedure):

    def __init__(self, args):
        super(Retriever, self).__init__(args)
        self._args = args


    def _init_model(self):
        self.vocab = Vocab.load(self._args['VOCAB_FILE'])
        self.src_vocab = self.vocab.src_vocab
        enc_embed_layer = nn.Embedding(len(self.src_vocab), int(self._args['--embed-size']), padding_idx=self.src_vocab[PADDING])
        self._model = Encoder(int(self._args['--embed-size']), int(self._args['--enc-hidden-size']), enc_embed_layer,
                              int(self._args['--num-layers']), float(self._args['--dropout-rate']))
        self._set_device()
        trained_params = torch.load(self._args['MODEL_PATH'], map_location=lambda storage, loc: storage)
        # select trained params
        new_params = {k[8:]: v for k, v in trained_params['state_dict'].items() if k.startswith('encoder')}
        # init params of encoder
        init_params = self._model.state_dict()
        # update params of encoder
        init_params.update(new_params)
        # load trained params for encoder
        self._model.load_state_dict(new_params)
        del self.vocab, trained_params, new_params, init_params

    def retrieve(self):
        """
        retrieve the most similar code based on code semantics, then save the source code and corresponding summaries.
        :param args:
        :return:
        """
        stop_word = [i for i in re.finditer("/", self._args['QUERY_OUT_PATH'])][-1].span()[1]
        test_vec_files = [file for file in os.listdir(self._args['QUERY_OUT_PATH'][: stop_word]) if file.startswith("test.vec.pkl.")]
        print(test_vec_files)
        # with open(, "rb") as fr:
        #     pkl.load(fr)



    def save_vecs(self):
        self._init_model()
        train_set, test_set = self.prepare_dataset()
        self.save_vec(train_set, self._args['SOURCE_OUT_PATH'])
        self.save_vec(test_set, self._args['QUERY_OUT_PATH'])

    def save_vec(self, data_set, out_path):
        vec_list = []
        for id, example in enumerate(tqdm(data_set)):
            vec = self.extract(example)
            vec_list.append(vec)
            if id != 0 and id % 4000 == 0 or id == len(data_set) - 1:
                with open(out_path+"."+str(id), "wb") as fw:
                    pkl.dump(vec_list, fw)
                vec_list = []

    def prepare_dataset(self):
        train_set = Dataset.create_from_file(self._args['TRAIN_SET_SRC'], self._args['TRAIN_SET_TGT'],
                                             self._args['--src-max-len'], self._args['--tgt-max-len'])
        test_set = Dataset.create_from_file(self._args['TEST_SET_SRC'], self._args['TEST_SET_TGT'],
                                            self._args['--src-max-len'], self._args['--tgt-max-len'])
        return train_set, test_set

    def extract(self, example):
        batch = Batch([example])
        src_tensor = batch.get_src_tensor(self.src_vocab, self._device)
        src_lens = batch.get_src_lens()
        src_encodings, _, _ = self._model(src_tensor, src_lens)
        pool = nn.MaxPool1d(max(src_lens), 1)
        r_c = pool(src_encodings.permute(0, 2, 1))
        return r_c

    def compare(self):
        pass

def main():
    args = docopt(__doc__)
    print(args)
    retriever = Retriever(args)
    # retriever.save_vecs()
    retriever.retrieve()


if __name__ == "__main__":
    main()
