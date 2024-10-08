#!/usr/bin/env python
#-*- coding:utf-8 -*-
# file name: semantic.py
# file description:
# created at:13/10/2021 4:23 PM

"""
Usage:
  semantic.py [options] MODEL_PATH VOCAB_FILE TRAIN_SET_SRC TRAIN_SET_TGT
  TEST_SET_SRC TEST_SET_TGT QUERY_OUT_PATH SOURCE_OUT_PATH SIMI_ID_OUT
  TEST_REF_SRC_1 TEST_REF_TGT_1

Options:
    -h --help                 show this screen.
    --cuda INT                use GPU [default: False]
    --embed-size INT          embed size [default: 256]
    --enc-hidden-size INT     encoder hidden size [default: 256]
    --num-layers INT          number of layers [default: 1]
    --dropout-rate FLOAT      dropout rate [default: 0.2]
    --src-max-len INT         max length of src [default: 152]
    --tgt-max-len INT         max length of tgt [default: 22]
"""

from modules.Encoder import Encoder
from docopt import docopt
from vocab import Vocab
from torch import nn
from modules.utils.common import *
from train import Procedure
from dataset import Dataset, Batch
from tqdm import tqdm
import pickle as pkl
import os
from multiprocessing import Pool
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing as mp

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

    def sub_retrieve(self, train_vec_files, train_stop_word, test_vec, test_file, ids, mg_lst):
        best = {"query_file":"", "query_id": -1, "source_file":"", "source_id": -1, "simi_score": -1}
        for train_file in train_vec_files:
            with open(self._args['SOURCE_OUT_PATH'][: train_stop_word] + train_file, "rb") as fr:
                train_vecs = pkl.load(fr)
            for idt, train_vec in enumerate(train_vecs):
                cur_score = round(cosine_similarity(test_vec, train_vec)[0][0], 2)
                if float(cur_score) > float(best['simi_score']):
                    best['query_file'] = test_file
                    best['query_id'] = ids
                    best['source_file'] = train_file
                    best['source_id'] = idt
                    best['simi_score'] = cur_score
        print("query {} completed!".format(ids))
        mg_lst.append(best)

    def retrieve(self):
        """
        retrieve the most similar code based on code semantics,
        then save their locations in order.
        :return:
        """

        test_stop_word = [i for i in re.finditer("/", self._args['QUERY_OUT_PATH'])][-1].span()[1]
        test_vec_files = [file for file in os.listdir(self._args['QUERY_OUT_PATH'][: test_stop_word]) if file.startswith("test.vec.pkl.")]
        train_stop_word = [i for i in re.finditer("/", self._args['SOURCE_OUT_PATH'])][-1].span()[1]
        train_vec_files = [file for file in os.listdir(self._args['SOURCE_OUT_PATH'][: train_stop_word]) if file.startswith("train.vec.pkl.")]

        mp_lst = mp.Manager().list()
        for test_file in test_vec_files:
            with open(self._args['QUERY_OUT_PATH'][: test_stop_word] + test_file, "rb") as fr:
                test_vecs = pkl.load(fr)
            p = Pool()
            for ids, test_vec in enumerate(test_vecs):
                p.apply_async(self.sub_retrieve, args=(train_vec_files, train_stop_word, test_vec, test_file, ids, mp_lst,))
            p.close()
            p.join()
        # sort
        mp_lst = sorted(mp_lst,key = lambda e:(float(e['query_file'].split(".")[-1]),int(e['query_id'])))
        with open(self._args['SIMI_ID_OUT'], "wb") as fw:
            pkl.dump(mp_lst, fw)

    def save_src_tgt(self):
        """
        save the retrieved sources and summaries.
        :return:
        """
        with open(self._args['SIMI_ID_OUT'], "rb") as fr:
            file = pkl.load(fr)
        train_set = Dataset.create_from_file(self._args['TRAIN_SET_SRC'], self._args['TRAIN_SET_TGT'],
                                             self._args['--src-max-len'], self._args['--tgt-max-len'])
        source = []
        summary = []
        for item in file:
            train_file = item['source_file']
            idt = item['source_id']

            source_file_id = int(train_file.split(".")[-1])
            if source_file_id % 4000 != 0:
                source_id = int(source_file_id / 4000) * 4000 + idt + 1
            else:
                source_id = source_file_id - 4000 + idt + 1
            source.append(" ".join(train_set[source_id].src_tokens))
            summary.append(" ".join(train_set[source_id].tgt_desc_tokens))

        with open(self._args['TEST_REF_SRC_1'], "w", encoding="utf-8") as fw:
            for line in source:
                fw.write(line+'\n')
        with open(self._args['TEST_REF_TGT_1'], "w", encoding="utf-8") as fw:
            for line in summary:
                fw.write(line+'\n')

    def save_vecs(self):
        """
        for the sake of time-saving, save samples' vectors in advance for further cosine similarity computing.
        :return:
        """
        self._init_model()
        train_set, test_set = self.prepare_dataset()
        self.save_vec(train_set, self._args['SOURCE_OUT_PATH'])
        self.save_vec(test_set, self._args['QUERY_OUT_PATH'])

    def save_vec(self, data_set, out_path):
        vec_list = []
        for id, example in enumerate(tqdm(data_set)):
            vec = self.extract(example).cpu().detach().numpy()
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
        # src_encoding (src_len, batch_size, hidden_size) -> (batch_size, hidden_size, src_len)
        src_encodings = src_encodings.permute(1, 2, 0)
        # 对src_len做pooling, 此外pooling只作用于最后一个维度，必要时需要置换维度
        pool = nn.MaxPool1d(max(src_lens))
        # eliminate the last dim
        r_c = pool(src_encodings).squeeze(2)
        return r_c

def main():
    import glob
    args = docopt(__doc__)
    print(args)
    retriever = Retriever(args)
    if len(glob.glob(args['QUERY_OUT_PATH']+".*")) == 0 or \
        len(glob.glob(args['SOURCE_OUT_PATH']+".*")) == 0:
        print("start to save encoded vectors for train and test set...")
        retriever.save_vecs()

    if not os.path.exists(args['SIMI_ID_OUT']):
        print("start to save the records of retrieved code...")
        retriever.retrieve()

    if not os.path.exists(args['TEST_REF_SRC_1']) or not os.path.exists(args['TEST_REF_TGT_1']):
        print("start to save sources and summaries...")
        retriever.save_src_tgt()

    print("Done!")

if __name__ == "__main__":
    main()
