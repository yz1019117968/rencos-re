#!/usr/bin/env python
#-*- coding:utf-8 -*-
# file name: Decoder.py
# file description:
# created at:29/9/2021 2:20 PM

from abc import ABC, abstractmethod
from typing import Any, Union, Callable, Tuple

from torch import DoubleTensor
from torch.nn import functional as F
from modules.utils.common import *
from dataset import Example
from modules.beam import Hypothesis
from modules.base import LSTMCell, Linear
from vocab import VocabEntry, BaseVocabEntry
from modules.GlobalAttention import GlobalAttention
from modules.utils.misc import negative_log_likelihood
from torch import nn, Tensor

class AbstractDecoder(nn.Module, ABC):
    def __init__(self):
        super(AbstractDecoder, self).__init__()
        # VocabEntry
        self.vocab = None
        self.embed_layer = None
        self.rnn_cell = None
        self.readout = None
        self.loss_func = None

    @property
    @abstractmethod
    def device(self):
        pass

    @abstractmethod
    def step(self, y_tm1_embed: Tensor, static_input: Any, state_tm1: Tuple) -> Tuple[Tuple, Tuple[Tensor]]:
        """
        :param y_tm1_embed:
        :param static_input:
        :param state_tm1:
        :return: state_tm1, out_vec
                 out_vec may contain all information to calculate words_log_prob
        """
        pass


class Decoder(AbstractDecoder, ABC):

    def __init__(self, embed_size: int, hidden_size: int, vocab: VocabEntry,
                 embed_layer: nn.Module, dropout_rate: float, args: dict,
                 loss_func: Callable = negative_log_likelihood):
        super(Decoder, self).__init__()
        self.input_feed = bool(args['--input-feed'])
        self.attn_type = str(args['--attn-type'])
        self.attn_func = str(args['--attn-func'])
        self.teacher_forcing_ratio = float(args['--teacher-forcing'])
        self.dropout_rate = dropout_rate
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.embed_layer = embed_layer
        # whether use input_feed
        if self.input_feed:
            input_size = embed_size + hidden_size
        else:
            input_size = embed_size
        self.attention = GlobalAttention(self.hidden_size, False, self.attn_type, self.attn_func)
        self.rnn_cell = LSTMCell(input_size, hidden_size, dropout=self.dropout_rate)
        # y.size+s_size+c_size
        self.generator_function = Linear(embed_size+hidden_size+hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    @property
    def device(self):
        return self.embed_layer.weight.device

    def prepare_prob_input(self, out_vecs: Union[Tuple[Tensor], List[Tuple[Tensor]]], **forward_kwargs) -> Tuple:
        """
        :param out_vecs: the out_vec of a step or the whole loop
        :return: inputs required by cal_words_log_prob
        """
        if isinstance(out_vecs, Tuple):
            return tuple([out_vecs[0]])
        elif isinstance(out_vecs, List):
            return tuple([torch.stack([out_v[0] for out_v in out_vecs])])
        else:
            raise Exception("Unexpected type of out_vecs: {}".format(type(out_vecs)))

    def cal_words_log_prob(self, att_ves: Tensor, *args) -> DoubleTensor:
        # (tgt_sent_len - 1, batch_size, tgt_vocab_size) or (batch_size, tgt_vocab_size)
        tgt_vocab_scores = self.readout(att_ves)
        words_log_prob = F.log_softmax(tgt_vocab_scores, dim=-1).double()
        return words_log_prob

    def cal_word_losses(self, target_tensor, words_log_prob: DoubleTensor) -> DoubleTensor:
        """
        :param target_tensor: (*tgt_len* - 1, batch_size) or (batch_size)!!!
        :param words_log_prob: logits
        :return: word_losses
        """
        # double for reproducability
        words_mask = (target_tensor != self.vocab[PADDING]).double()
        # (tgt_sent_len - 1, batch_size)
        word_losses = self.loss_func(words_log_prob, target_tensor, words_mask)
        return word_losses

    def _init_step(self, **forward_args):
        h_tm1 = forward_args['dec_init_state']
        # (batch_size, hidden_size)
        att_tm1 = torch.zeros(forward_args['src_encodings'].size(0), self.hidden_size, device=self.device)
        state_tm1 = (h_tm1, att_tm1)
        return state_tm1

    def step(self, y_tm1_embed: Tensor, static_input: Any, state_tm1: Tuple) -> Tuple[Tuple, Tuple]:
        src_encodings, src_sent_masks = static_input
        # h_tm1 (h_t-1, c_t-1);
        h_tm1, att_tm1 = state_tm1
        if self.input_feed:
            x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
        else:
            x = torch.cat([y_tm1_embed], dim=-1)
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = self.rnn_cell(x, h_tm1)
        # assert False, "STOP"
        # ctx_t: src_encoding_state
        # todo src_sent_masks need to be changed to sentence lengths
        att_t, alpha_t = self.attention(
            h_t, src_encodings, src_sent_masks)
        cat_final = torch.cat([y_tm1_embed, h_t, att_t], dim=-1)
        decoder_output = self.tanh(self.generator_function(cat_final))

        state_tm1 = ((h_t, cell_t), att_t)
        return state_tm1, (decoder_output, alpha_t)

    def forward(self, tgt_in_tensor: Tensor, tgt_out_tensor: Tensor, src_encodings, src_lens, src_last_state, src_last_cell) \
            -> Tuple[Tensor, Tensor]:
        att_tm1 = torch.zeros(src_encodings.size(0), self.hidden_size, device=self.device)
        teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if teacher_forcing:
            out_vecs = []
            print("tgt_in_tensor: ", tgt_in_tensor.shape)
            tgt_in_tensor = tgt_in_tensor.permute(1, 0)
            tgt_in_embeddings = self.embed_layer(tgt_in_tensor).permute(1, 0, 2)
            # start from y_0=`<s>`, iterate until y_{T-1}
            for y_tm1_embed in tgt_in_embeddings.split(split_size=1, dim=0):
                # (batch_size, embed_size)
                y_tm1_embed = y_tm1_embed.squeeze(0)
                print(y_tm1_embed .shape)
                assert False, "STOP"
                # out_vec may contain tensors related to attentions
                state_tm1, out_vec = self.step(y_tm1_embed, (src_encodings, src_lens), ((src_last_state, src_last_cell), att_tm1))
                out_vecs.append(out_vec)
            # (tgt_in_sent_len - 1, batch_size, hidden_size)
            prob_input = self.prepare_prob_input(out_vecs, **kwargs)
            words_log_prob = self.cal_words_log_prob(*prob_input)
            ys = words_log_prob.max(dim=-1)[1]
        else:
            words_log_prob = []
            ys = []
            y_t = tgt_in_tensor[0]
            for di in range(tgt_in_tensor.size(0)):
                out_of_vocab = (y_t >= len(self.vocab))
                y_tm1 = y_t.masked_fill(out_of_vocab, self.vocab.unk_id)

                # (batch_size, embed_size)
                y_tm1_embed = self.embed_layer(y_tm1)
                # out_vec may contain tensors related to attentions
                state_tm1, out_vec = self.step(y_tm1_embed, static_input, state_tm1)

                prob_input = self.prepare_prob_input(out_vec, **kwargs)
                # (batch_size, vocab_size)
                log_prob_t = self.cal_words_log_prob(*prob_input)
                words_log_prob.append(log_prob_t)
                y_t = log_prob_t.max(dim=1)[1]
                ys.append(y_t)
            words_log_prob = torch.stack(words_log_prob, dim=0)
            ys = torch.stack(ys, dim=0)

        word_losses = self.cal_word_losses(tgt_out_tensor, words_log_prob)

        return word_losses, ys

    def beam_search(self, example: Example, beam_size: int, max_dec_step: int, BeamClass, **kwargs) -> List[Hypothesis]:
        """
        NOTE: the batch size must be 1
        """
        vocab = self.get_decode_vocab(example)
        static_input = self._init_loop(**kwargs)
        state_tm1 = self._init_step(**kwargs)

        # should not use self.vocab directly
        beam = BeamClass(vocab, self.device, beam_size, example.src_tokens)

        cur_step = 0
        while (not beam.is_finished) and cur_step < max_dec_step:
            cur_step += 1
            y_tm1 = beam.next_y_tm1()
            y_tm1_embed = self.embed_layer(y_tm1)
            cur_static_input = beam.expand_static_input(static_input)
            # assert False, "FALSE"
            state_tm1, out_vec = self.step(y_tm1_embed, cur_static_input, state_tm1)
            prob_input = self.prepare_prob_input(out_vec, **kwargs)
            words_log_prob = self.cal_words_log_prob(*prob_input)
            state_tm1 = beam.step(words_log_prob, state_tm1)

        return beam.get_final_hypos()
