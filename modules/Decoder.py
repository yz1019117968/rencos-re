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
from modules.base import Linear
from torch.nn import LSTM
from vocab import VocabEntry, BaseVocabEntry
from modules.GlobalAttention import GlobalAttention
from modules.utils.misc import negative_log_likelihood
from torch import nn, Tensor


class Decoder(nn.Module, ABC):

    def __init__(self, embed_size: int, hidden_size: int, vocab: VocabEntry,
                 embed_layer: nn.Module, num_layers: int, dropout_rate: float, args: dict,
                 loss_func: Callable = negative_log_likelihood):
        super(Decoder, self).__init__()
        self.input_feed = bool(args['--input-feed'])
        self.attn_type = str(args['--attn-type'])
        self.attn_func = str(args['--attn-func'])
        self.teacher_forcing_ratio = float(args['--teacher-forcing'])
        self.dropout_rate = dropout_rate
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab = vocab
        self.embed_layer = embed_layer
        # whether use input_feed
        if self.input_feed:
            input_size = embed_size + hidden_size
        else:
            input_size = embed_size
        self.attention = GlobalAttention(self.hidden_size, False, self.attn_type, self.attn_func)
        self.rnn_layer = LSTM(input_size, self.hidden_size, num_layers, bidirectional=False, batch_first=False,
                              dropout=dropout_rate)
        # y.size+s_size+c_size
        self.generator_function = Linear(embed_size+hidden_size+hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.readout = Linear(self.hidden_size, len(self.vocab), bias=False)
        self.loss_func = loss_func

    @property
    def device(self):
        return self.embed_layer.weight.device

    def prepare_prob_input(self, out_vecs: Union[Tensor, List[Tensor]]) -> Tuple:
        """
        :param out_vecs: the out_vec of a step or the whole loop
        :return: inputs required by cal_words_log_prob, type: tuple
        """
        if isinstance(out_vecs, Tensor):
            return tuple([out_vecs])
        elif isinstance(out_vecs, List):
            return tuple([torch.stack([out_v[0] for out_v in out_vecs])])
        else:
            raise Exception("Unexpected type of out_vecs: {}".format(type(out_vecs)))

    def cal_words_log_prob(self, att_ves: Tensor) -> DoubleTensor:
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

    def step(self, y_tm1_embed, att_tm1, src_encodings, src_lens, last_state, last_cell):
        if self.input_feed:
            x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
        else:
            x = torch.cat([y_tm1_embed], dim=-1)
        out_vec, (last_state, last_cell) = self.rnn_layer(x, (last_state, last_cell))
        # for each variable, remember to put in on gpu
        att_t, alpha_t = self.attention(
            last_state.permute(1, 0, 2), src_encodings,
            src_lens
        )
        cat_final = torch.cat([y_tm1_embed, last_state, att_t], dim=-1)
        decoder_output = self.tanh(self.generator_function(cat_final))
        return att_t, alpha_t, last_state, last_cell, decoder_output

    def forward(self, tgt_in_tensor: Tensor, tgt_out_tensor: Tensor, src_encodings, src_lens, last_state, last_cell) \
            -> Tuple[Tensor, Tensor]:
        """
        :param tgt_in_tensor: (tgt_len - 1, batch_size)
        :param tgt_out_tensor: (tgt_len - 1, batch_size)
        :param src_encodings: (batch_size, src_len, hidden_size)
        :param src_lens: (batch_size)
        :param last_state: (1, batch_size, hidden_size)
        :param last_cell: (1, batch_size, hidden_size)
               att_tm1:(1, batch_size, hidden_size)
        :return:
        """
        att_tm1 = torch.zeros(1, tgt_in_tensor.size(1), self.hidden_size, device=self.device)
        teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        src_lens = torch.tensor(src_lens, dtype=torch.int).to(self.device)
        out_vecs = []
        att_maps = []
        if teacher_forcing:
            tgt_in_tensor = tgt_in_tensor.permute(1, 0)
            tgt_in_embeddings = self.embed_layer(tgt_in_tensor).permute(1, 0, 2)
            # start from y_0=`<s>`, iterate until y_{T-1}
            for y_tm1_embed in tgt_in_embeddings.split(split_size=1, dim=0):
                # if self.input_feed:
                #     x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
                # else:
                #     x = torch.cat([y_tm1_embed], dim=-1)
                # out_vec, (last_state, last_cell) = self.rnn_layer(x, (last_state, last_cell))
                # # for each variable, remember to put in on gpu
                # att_t, alpha_t = self.attention(
                #     last_state.permute(1, 0, 2), src_encodings,
                #     torch.tensor(src_lens, dtype=torch.int).to(self.device)
                # )
                # cat_final = torch.cat([y_tm1_embed, last_state, att_t], dim=-1)
                # decoder_output = self.tanh(self.generator_function(cat_final))
                att_tm1, alpha_t, last_state, last_cell, decoder_output = self.step(y_tm1_embed, att_tm1, src_encodings, src_lens, last_state, last_cell)
                out_vecs.append(decoder_output)
                att_maps.append(alpha_t)
            # out_vecs: (tgt_in_sent_len - 1, batch_size, hidden_size)
            prob_input = self.prepare_prob_input(out_vecs)
            words_log_prob = self.cal_words_log_prob(*prob_input)
            ys = words_log_prob.max(dim=-1)[1]
        else:
            raise Exception("Decay Sampling has not been implemented yet! Pls set the teacher forcing rate to 1.0.")
            # todo decay sampling
            # words_log_prob = []
            # ys = []
            # y_t = tgt_in_tensor[0]
            # for di in range(tgt_in_tensor.size(0)):
            #     out_of_vocab = (y_t >= len(self.vocab))
            #     y_tm1 = y_t.masked_fill(out_of_vocab, self.vocab.unk_id)
            #
            #     # (batch_size, embed_size)
            #     y_tm1_embed = self.embed_layer(y_tm1)
            #     # out_vec may contain tensors related to attentions
            #     state_tm1, out_vec = self.step(y_tm1_embed, static_input, state_tm1)
            #
            #     prob_input = self.prepare_prob_input(out_vec, **kwargs)
            #     # (batch_size, vocab_size)
            #     log_prob_t = self.cal_words_log_prob(*prob_input)
            #     words_log_prob.append(log_prob_t)
            #     y_t = log_prob_t.max(dim=1)[1]
            #     ys.append(y_t)
            # words_log_prob = torch.stack(words_log_prob, dim=0)
            # ys = torch.stack(ys, dim=0)

        word_losses = self.cal_word_losses(tgt_out_tensor, words_log_prob)

        return word_losses, ys

    def beam_search(self, example: Example, beam_size: int, max_dec_step: int, BeamClass,
                    src_encodings, src_lens, last_state, last_cell) -> List[Hypothesis]:
        """
        NOTE: the batch size must be 1
        """
        beam = BeamClass(self.vocab, self.device, beam_size, example.src_tokens)
        cur_step = 0
        att_tm1 = torch.zeros(1, 1, self.hidden_size, device=self.device)
        src_lens = torch.tensor(src_lens, dtype=torch.int).to(self.device)
        while (not beam.is_finished) and cur_step < max_dec_step:
            cur_step += 1
            y_tm1 = beam.next_y_tm1()
            y_tm1_embed = self.embed_layer(y_tm1).permute(1, 0, 2)
            cur_static_input = beam.expand_static_input((src_encodings, src_lens))
            # assert False, "FALSE"
            att_t, alpha_t, state_t, cell_t, decoder_output = self.step(y_tm1_embed, att_tm1, *cur_static_input, last_state, last_cell)
            prob_input = self.prepare_prob_input(decoder_output)
            words_log_prob = self.cal_words_log_prob(*prob_input)
            beam.step(words_log_prob)
            (state_t, att_t) = beam.expand_iter_input(((state_t.permute(1, 0, 2), cell_t.permute(1, 0, 2)), att_t.permute(1, 0, 2)))
            last_state, last_cell = state_t[0].permute(1, 0, 2), state_t[1].permute(1,0,2)
            att_tm1 = att_t.permute(1, 0, 2)
        return beam.get_final_hypos()

    def rencos_beam_search(self, example: Example, beam_size: int, max_dec_step: int, BeamClass, src_encodings, src_lens, last_state, last_cell,
                           src_encodings_0, src_lens_0, last_state_0, last_cell_0, src_encodings_1, src_lens_1, last_state_1, last_cell_1,
                           prs_0: List, prs_1: List, _lambda):
        beam = BeamClass(self.vocab, self.device, beam_size, example.src_tokens)
        cur_step = 0
        att_tm1 = torch.zeros(1, 1, self.hidden_size, device=self.device)
        att_tm1_0 = torch.zeros(1, 1, self.hidden_size, device=self.device)
        att_tm1_1 = torch.zeros(1, 1, self.hidden_size, device=self.device)

        src_lens = torch.tensor(src_lens, dtype=torch.int).to(self.device)
        src_lens_0 = torch.tensor(src_lens_0, dtype=torch.int).to(self.device)
        src_lens_1 = torch.tensor(src_lens_1, dtype=torch.int).to(self.device)
        while (not beam.is_finished) and cur_step < max_dec_step:
            y_tm1 = beam.next_y_tm1()
            y_tm1_embed = self.embed_layer(y_tm1).permute(1, 0, 2)

            cur_static_input = beam.expand_static_input((src_encodings, src_lens))
            cur_static_input_0 = beam.expand_static_input((src_encodings_0, src_lens_0))
            cur_static_input_1 = beam.expand_static_input((src_encodings_1, src_lens_1))

            att_t, alpha_t, state_t, cell_t, decoder_output = self.step(y_tm1_embed, att_tm1, *cur_static_input, last_state, last_cell)
            att_t_0, alpha_t_0, state_t_0, cell_t_0, decoder_output_0 = self.step(y_tm1_embed, att_tm1_0, *cur_static_input_0, last_state_0, last_cell_0)
            att_t_1, alpha_t_1, state_t_1, cell_t_1, decoder_output_1 = self.step(y_tm1_embed, att_tm1_1, *cur_static_input_1, last_state_1, last_cell_1)
            prob_input = self.prepare_prob_input(decoder_output)
            prob_input_0 = self.prepare_prob_input(decoder_output_0)
            prob_input_1 = self.prepare_prob_input(decoder_output_1)

            words_log_prob = self.cal_words_log_prob(*prob_input)
            words_log_prob_0 = self.cal_words_log_prob(*prob_input_0)
            words_log_prob_1 = self.cal_words_log_prob(*prob_input_1)

            _words_log_prob = words_log_prob + _lambda * prs_0[cur_step] * words_log_prob_0 + _lambda * prs_1[cur_step] * words_log_prob_1
            beam.step(_words_log_prob)

            (state_t, att_t) = beam.expand_iter_input(((state_t.permute(1, 0, 2), cell_t.permute(1, 0, 2)), att_t.permute(1, 0, 2)))
            last_state, last_cell = state_t[0].permute(1, 0, 2), state_t[1].permute(1,0,2)
            att_tm1 = att_t.permute(1, 0, 2)

            (state_t_0, att_t_0) = beam.expand_iter_input(((state_t_0.permute(1, 0, 2), cell_t_0.permute(1, 0, 2)), att_t_0.permute(1, 0, 2)))
            last_state_0, last_cell_0 = state_t_0[0].permute(1, 0, 2), state_t_0[1].permute(1,0,2)
            att_tm1_0 = att_t_0.permute(1, 0, 2)

            (state_t_1, att_t_1) = beam.expand_iter_input(((state_t_1.permute(1, 0, 2), cell_t_1.permute(1, 0, 2)), att_t_1.permute(1, 0, 2)))
            last_state_1, last_cell_1 = state_t_1[0].permute(1, 0, 2), state_t_1[1].permute(1,0,2)
            att_tm1_1 = att_t_1.permute(1, 0, 2)
            cur_step += 1
        return beam.get_final_hypos()
