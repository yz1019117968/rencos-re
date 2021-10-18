#!/usr/bin/env python
#-*- coding:utf-8 -*-
# file name: beam.py
# file description:
# created at:7/10/2021 10:25 AM

from collections import namedtuple
from typing import Tuple

from abc import abstractmethod, ABC
from torch import Tensor
from modules.utils.common import *
from vocab import BaseVocabEntry

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class AbstractBeam(ABC):
    @property
    @abstractmethod
    def is_finished(self):
        pass

    @abstractmethod
    def next_y_tm1(self):
        pass

    @abstractmethod
    def expand_static_input(self, static_input: Tuple[Tensor]) -> Tuple[Tensor]:
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_final_hypos(self):
        pass


class BaseBeam(AbstractBeam, ABC):
    def __init__(self):
        self.completed_hypos = None
        self.live_hypo_sents = None
        self.live_hypo_scores = None
        self.beam_size = None

    @property
    def is_finished(self):
        return len(self.completed_hypos) >= self.beam_size or len(self.live_hypo_sents) == 0

    def expand_static_input(self, static_input: Tuple[Tensor]) -> Tuple[Tensor]:
        new_static_input = []
        for s_input in static_input:
            # make sure that len(self.live_hypo_sents) will not be zero
            new_size = [len(self.live_hypo_sents)] + [1] * (s_input.dim() - 1)
            new_static_input.append(s_input.repeat(new_size))
        return tuple(new_static_input)

    def expand_iter_input(self, state_tm1, new_hypo_ids):
        new_state = []
        for s in state_tm1:
            if isinstance(s, tuple):
                new_state.append(tuple(sub_s[new_hypo_ids] for sub_s in s))
            else:
                new_state.append(s[new_hypo_ids])
        return tuple(new_state)

    def get_final_hypos(self):
        if len(self.completed_hypos) == 0:
            if len(self.live_hypo_sents) == 0:
                self.completed_hypos.append(Hypothesis(value=[], score=0))
            else:
                self.add_completed_hypo(0, self.live_hypo_scores[0].item())
        # 最终5个句子的排名根据句子长短和分数共同评价
        self.completed_hypos.sort(key=lambda hypo: hypo.score / max(len(hypo.value), 1), reverse=True)
        return self.completed_hypos

    @abstractmethod
    def add_completed_hypo(self, hypo_id, score):
        pass


class Beam(BaseBeam):
    def __init__(self, vocab: BaseVocabEntry, device: torch.device, beam_size: int):
        super().__init__()
        self.vocab = vocab
        self.device = device
        self.beam_size = beam_size
        self.completed_hypos = []
        # store the word ids of each sent
        self.live_hypo_sents = [[self.vocab[TGT_START]]]
        # store the sum(log_prob(word))
        self.live_hypo_scores = torch.zeros(len(self.live_hypo_sents), dtype=torch.float64, device=self.device)

    def _topk_candidates(self, scores: Tensor, k: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param scores: (..., batch_size, vocab_size)
        :param k: top-k
        :return: topk_scores, hypo_ids, word_ids
        """
        vocab_size = scores.size(-1)
        topk_scores, topk_positions = torch.topk(scores.view(-1), k, sorted=True)
        hypo_ids = topk_positions // vocab_size
        word_ids = topk_positions % vocab_size
        return topk_scores, hypo_ids, word_ids

    def next_y_tm1(self):
        # NOTE: for ext vocab, we need to clip the y to the base_vocab
        live_hypo_words = [sent[-1] for sent in self.live_hypo_sents]
        y_tm1 = torch.tensor(live_hypo_words, dtype=torch.long, device=self.device).unsqueeze(1)
        return y_tm1

    def step(self, words_log_prob, *args, **kwargs):
        # 因为是log_prob, 所以为负
        cur_hypo_scores = self.live_hypo_scores.unsqueeze(1).expand_as(words_log_prob) + words_log_prob
        # prepare more candidates to avoid empty candidate
        topk_scores, topk_hypo_ids, topk_word_ids = self._topk_candidates(cur_hypo_scores,
                                                                          self.beam_size - len(
                                                                              self.completed_hypos) + 5)
        new_hypo_sents = []
        new_hypo_scores = []
        new_hypo_ids = []
        for score, hypo_id, word_id in zip(topk_scores, topk_hypo_ids, topk_word_ids):
            if len(self.completed_hypos) >= self.beam_size or len(new_hypo_sents) >= self.beam_size:
                break
            score, hypo_id, word_id = score.item(), hypo_id.item(), word_id.item()
            if word_id == self.vocab[TGT_END]:
                self.add_completed_hypo(hypo_id, score)
                continue
            new_hypo_sents.append(self.live_hypo_sents[hypo_id] + [word_id])
            new_hypo_scores.append(score)
            new_hypo_ids.append(hypo_id)
        self.live_hypo_sents = new_hypo_sents
        self.live_hypo_scores = torch.tensor(new_hypo_scores, dtype=torch.float64, device=self.device)
        # update the states
        # new_state = []
        # for s in state_tm1:
        #     if isinstance(s, tuple):
        #         new_state.append(tuple(sub_s[new_hypo_ids] for sub_s in s))
        #     else:
        #         new_state.append(s[new_hypo_ids])
        # return tuple(new_state)

    def add_completed_hypo(self, hypo_id, score):
        # convert word_id to word
        hypo_sent = self.vocab.indices2words(self.live_hypo_sents[hypo_id][1:])
        self.completed_hypos.append(Hypothesis(value=hypo_sent, score=score))
