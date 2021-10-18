#!/usr/bin/env python
#-*- coding:utf-8 -*-
# file name: infer.py
# file description:
# created at:11/10/2021 10:35 AM

"""
Usage:
    infer.py [options] MODEL_PATH TEST_SET_SRC TEST_SET_TGT OUTPUT_FILE OUTPUT_FILE_OBS

Options:
    -h --help                   show this screen.
    --cuda INT                  use GPU [default: true]
    --rencos BOOL               whether execute the infer for rencos [default: False]
    --test-ref-src-0 FILE       test.ref.src.0 [default: ./test.ref.src.0]
    --test-ref-src-1 FILE       test.ref.src.1 [default: ./test.ref.src.1]
    --test-ref-tgt-0 FILE       ast.out [default: ./ast.out]
    --test-ref-tgt-1 FILE       test.ref.tgt.1 [default: ./test.ref.tgt.1]
    --src-max-len INT           max length of src [default: 100]
    --tgt-max-len INT           max length of tgt [default: 50]
    --prs-0 FILE                sim(c_test, c_ref_0) [default: ./prs.0]
    --prs-1 FILE                sim(c_test, c_ref_1) [default: ./prs.1]
    --lambda INT                _lambda [default: 3]
    --model-class STR           model class [default: modules.Seq2Seq.Seq2Seq]
    --seed INT                  random seed [default: 0]
    --beam-size INT             beam size [default: 5]
    --max-dec-step INT          max decode steps [default: 50]
    --beam-class STR            beam class used for beam search [default: modules.beam.Beam]
"""

from typing import Callable

import torch
import json
import logging
from tqdm import tqdm
from docopt import docopt
from dataset import Dataset, Example
from modules.beam import Hypothesis
from modules.utils.common import get_attr_by_name, set_reproducibility
from train import Procedure, List

logging.basicConfig(level=logging.INFO)


class Infer(Procedure):
    def __init__(self, args: dict):
        super(Infer, self).__init__(args)

    @property
    def beam_size(self):
        return int(self._args['--beam-size'])

    @property
    def max_dec_step(self):
        return int(self._args['--max-dec-step'])

    def _init_model(self):
        model_class = get_attr_by_name(self._args['--model-class'])
        self._model = model_class.load(self._args['MODEL_PATH'])
        self._set_device()

    def beam_search(self, data_set):
        logging.info("Using beam class: " + self._args['--beam-class'])
        BeamClass = get_attr_by_name(self._args['--beam-class'])
        was_training = self._model.training
        self._model.eval()

        hypos = []
        with torch.no_grad():
            for example in tqdm(data_set):
                example_hypos = self._model.beam_search(example, self.beam_size, self.max_dec_step, BeamClass)
                # print("example_hypos: ", example_hypos)
                hypos.append(example_hypos)

        if was_training:
            self._model.train()

        return hypos

    def infer(self) -> List[List[Hypothesis]]:
        test_set = Dataset.create_from_file(self._args['TEST_SET_SRC'], self._args['TEST_SET_TGT'],
                                            self._args['--src-max-len'], self._args['--tgt-max-len'])
        self._init_model()
        hypos = self.beam_search(test_set)

        with open(self._args['OUTPUT_FILE'], 'w') as f:
            json.dump(hypos, f)

        with open(self._args['OUTPUT_FILE_OBS'], 'w', encoding="utf-8") as fw:
            for hypo, example in zip(hypos,test_set):
                fw.write(f'{example._sample_id}: ' + " ".join(example.tgt_desc_tokens))
                fw.write('\n')
                for i in hypo:
                    fw.write(" ".join(i[0]))
                    fw.write('\n')
                fw.write('\n\n')
        return hypos

    def prepare_prs(self):
        prs_0 = []
        with open(self._args['--prs-0'], "r", encoding="utf-8") as fr:
            for i in fr.readlines():
                prs_0.append(float(i.strip()))
        prs_1 = []
        with open(self._args['--prs-1'], "r", encoding="utf-8") as fr:
            for i in fr.readlines():
                prs_1.append(float(i.strip()))
        return prs_0, prs_1

    def beam_search_rencos(self, test_set, test_set_0, test_set_1):
        logging.info("Using beam class: " + self._args['--beam-class']+ "to execute rencos.")
        BeamClass = get_attr_by_name(self._args['--beam-class'])
        was_training = self._model.training
        self._model.eval()
        prs_0, prs_1 = self.prepare_prs()
        hypos = []
        with torch.no_grad():
            for (example, example_0, example_1) in tqdm(zip(test_set, test_set_0, test_set_1), total=len(test_set)):
                example_hypos = self._model.rencos(example, example_0, example_1, self.beam_size, self.max_dec_step,
                    prs_0, prs_1, float(self._args['--lambda']), BeamClass)
                # print("example_hypos: ", example_hypos)
                hypos.append(example_hypos)

        if was_training:
            self._model.train()

        return hypos

    def infer_rencos(self):
        test_set = Dataset.create_from_file(self._args['TEST_SET_SRC'], self._args['TEST_SET_TGT'],
                                            self._args['--src-max-len'], self._args['--tgt-max-len'])
        test_set_0 = Dataset.create_from_file(self._args['--test-ref-src-0'], self._args['--test-ref-tgt-0'],
                                            self._args['--src-max-len'], self._args['--tgt-max-len'])
        test_set_1 = Dataset.create_from_file(self._args['--test-ref-src-1'], self._args['--test-ref-tgt-1'],
                                            self._args['--src-max-len'], self._args['--tgt-max-len'])

        self._init_model()
        hypos = self.beam_search_rencos(test_set, test_set_0, test_set_1)
        with open(self._args['OUTPUT_FILE'], 'w') as f:
            json.dump(hypos, f)

        with open(self._args['OUTPUT_FILE_OBS'], 'w', encoding="utf-8") as fw:
            for hypo, example in zip(hypos,test_set):
                fw.write(f'{example._sample_id}: ' + " ".join(example.tgt_desc_tokens))
                fw.write('\n')
                for i in hypo:
                    fw.write(" ".join(i[0]))
                    fw.write('\n')
                fw.write('\n\n')
        return hypos


def infer(args):
    if args['--seed'] is not None:
        seed = int(args['--seed'])
        set_reproducibility(seed)

    infer_instance = Infer(args)
    if not args['--rencos']:
        infer_instance.infer()
    else:
        infer_instance.infer_rencos()


def main():
    args = docopt(__doc__)
    print(args)
    infer(args)


if __name__ == '__main__':
    main()
