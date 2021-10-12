#!/usr/bin/env python
#-*- coding:utf-8 -*-
# file name: eval.py.py
# file description:
# created at:11/10/2021 7:59 PM

"""
Usage:
    eval.py [options] TEST_SET_SRC TEST_SET_TGT RESULT_FILE

Options:
    -h --help                   show this screen.
    --src-max-len INT           max length of src [default: 100]
    --tgt-max-len INT           max length of tgt [default: 50]
    --metrics LIST              metrics to calculate [default: sent_bleu,corp_bleu,rouge,meteor]
    --eval-class STR            the class used to evaluate [default: Evaluator]
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple
from docopt import docopt
from dataset import Dataset
logging.basicConfig(level=logging.INFO)
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import tensorflow as tf
import numpy as np
from rouge import Rouge


class BaseMetric(ABC):
    @abstractmethod
    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]], *args, **kwargs) -> float:
        """
        :param hypos: each hypo contains k sents, for accuracy, only use the first sent, for recall, use k sents
        :param references: the dst desc sents
        :param kwargs:
        :return:
        """
        pass


class SentBLEU(BaseMetric):
    def __init__(self):
        super(SentBLEU, self).__init__()
        self.chencherry = SmoothingFunction()

    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]], *args, **kwargs) -> float:
        scores = []
        for hypo, ref in zip(hypos, references):
            """
            :param references: reference sentences
            :type references: list(list(str))
            :param hypothesis: a hypothesis sentence
            :type hypothesis: list(str)
            """
            scores.append(sentence_bleu([ref], hypo[0], weights=(0.25, 0.25, 0.25, 0.25),
                                        smoothing_function=self.chencherry.method1))
        return np.mean(scores)

class CorpBLEU(BaseMetric):
    def __init__(self):
        super(CorpBLEU, self).__init__()
        self.chencherry = SmoothingFunction()

    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]], *args, **kwargs) -> float:
        fine_hypos = [hypo[0] for hypo in hypos]
        fine_refs = [[ref] for ref in references]
        """
        :param list_of_references: a corpus of lists of reference sentences, w.r.t. hypotheses
        :type list_of_references: list(list(list(str)))
        :param hypotheses: a list of hypothesis sentences
        :type hypotheses: list(list(str))
        """
        return corpus_bleu(fine_refs, fine_hypos, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.chencherry.method1)

class ROUGE(BaseMetric):
    def __init__(self):
        super(ROUGE, self).__init__()
        self.rouge = Rouge()

    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]], *args, **kwargs) -> float:
        scores = []
        for hypo, ref in zip(hypos, references):
            """
            :param references: reference sentences
            :type references: string
            :param hypothesis: a hypothesis sentence
            :type hypothesis: string
            """
            scores.append(self.rouge.get_scores(" ".join(hypo[0]), " ".join(ref))[0]['rouge-l']['f'])
        return np.mean(scores)

class METEOR(BaseMetric):
    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]], *args, **kwargs) -> float:
        scores = []
        for hypo, ref in zip(hypos, references):
            """
            :param references: reference sentences
            :type references: list(str)
            :param hypothesis: a hypothesis sentence
            :type hypothesis: str
            """
            scores.append(meteor_score([" ".join(ref)], " ".join(hypo[0])))
        return np.mean(scores)

class BaseEvaluator(ABC):
    @abstractmethod
    def load_hypos_and_refs(self) -> Tuple[List[List[List[str]]], List[List[str]], List[List[str]]]:
        pass


class Evaluator(BaseEvaluator):
    METRIC_MAP = {
        "sent_bleu": SentBLEU(),
        "corp_bleu": CorpBLEU(),
        "rouge": ROUGE(),
        "meteor": METEOR()
    }

    def __init__(self, args: dict, metric_map: dict = None):
        self.args = args
        self.metric_map = metric_map if metric_map else self.METRIC_MAP

    def load_hypos(self) -> List[List[List[str]]]:
        with open(self.args['RESULT_FILE'], 'r') as f:
            results = json.load(f)
        return self.load_hypos_raw(results)

    def load_hypos_raw(self, results) -> List[List[List[str]]]:
        # only use the first hypo
        assert type(results[0][0][0]) == list and type(results[0][0][1] == float), \
            "Each example should have a list of Hypothesis. Please prepare your result like " \
            "[Hypothesis(desc, score), ...]"
        # NOTE: results: List[List[list of tokens]]
        hypos = [[hypo[0] for hypo in r] for r in results]
        return hypos

    def load_hypos_and_refs(self):
        test_set = Dataset.create_from_file(self.args['TEST_SET_SRC'], self.args['TEST_SET_TGT'],
                                            self.args['--src-max-len'], self.args['--tgt-max-len'])
        references = list(test_set.get_ground_truth())
        hypos = self.load_hypos()
        return hypos, references

    def cal_metrics(self, metrics: Iterable[str], hypos: List[List[List[str]]], references: List[List[str]]):
        results = {}
        for metric in metrics:
            instance = self.metric_map[metric.lower()]
            results[metric] = instance.eval(hypos, references)
        return results

    def evaluate(self):
        metrics = self.args['--metrics'].split(',')
        hypos, references = self.load_hypos_and_refs()
        assert type(hypos[0][0]) == type(references[0])
        results = self.cal_metrics(metrics, hypos, references)
        logging.info(results)
        print(results)
        return results


def evaluate(args):
    EvalClass = globals()[args['--eval-class']]
    evaluator = EvalClass(args)
    return evaluator.evaluate()


def main():
    args = docopt(__doc__)
    evaluate(args)


if __name__ == '__main__':
    main()
