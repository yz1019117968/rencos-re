#!/usr/bin/env python
#-*- coding:utf-8 -*-
# file name: Seq2Seq.py
# file description:
# created at:4/10/2021 5:26 PM

from torch import nn


import logging
from modules import base
from modules.beam import Beam
from dataset import Batch
from vocab import Vocab
from modules.Decoder import *
from modules.Encoder import Encoder
from modules.Decoder import Decoder
from modules.utils.misc import negative_log_likelihood

logging.basicConfig(level=logging.INFO)

class Seq2Seq(base.BaseModel, ABC):
    @staticmethod
    def log_args(**kwargs):
        logging.info("Create model using parameters:")
        for key, value in kwargs.items():
            logging.info("{}={}".format(key, value))

    @staticmethod
    def prepare_model_params(args):
        return int(args['--embed-size']), int(args['--num-layers']), int(args['--enc-hidden-size']), \
               int(args['--dec-hidden-size']), Vocab.load(args['VOCAB_FILE']), float(args['--dropout-rate']), args

    def __init__(self, embed_size: int, num_layers: int, enc_hidden_size: int, dec_hidden_size: int,
                 vocab: Vocab, dropout_rate, args: dict,
                 loss_func: Callable = negative_log_likelihood):
        super(Seq2Seq, self).__init__()
        self.init_embeddings(vocab, embed_size)
        self.encoder = Encoder(embed_size, enc_hidden_size, self.enc_embed_layer, num_layers, dropout_rate)
        self.decoder = Decoder(embed_size, dec_hidden_size, self.tgt_vocab, self.dec_embed_layer, num_layers, dropout_rate, args, loss_func)

    def init_embeddings(self, vocab: 'Vocab', embed_size):
        self.src_vocab = vocab.src_vocab
        self.tgt_vocab = vocab.tgt_vocab
        self.enc_embed_layer = nn.Embedding(len(self.src_vocab), embed_size, padding_idx=self.src_vocab[PADDING])
        self.dec_embed_layer = nn.Embedding(len(self.tgt_vocab), embed_size, padding_idx=self.tgt_vocab[PADDING])

    @property
    def device(self) -> torch.device:
        return self.encoder.embed_layer.weight.device

    def _prepare_dec_init_state(self, last_cells: List[Tensor]):
        """
        :param last_cells: List[(batch_size, edit_vec_size)]
        :return:
            (batch_size, hidden_size)
            (batch_size, hidden_size)
        """
        # (batch_size, hidden_size)
        dec_init_cell = self.dec_state_init(torch.cat(last_cells, dim=-1))
        dec_init_state = torch.tanh(dec_init_cell)
        return dec_init_state, dec_init_cell

    def forward(self, batch: Batch) -> Tensor:
        src_tensor = batch.get_src_tensor(self.src_vocab, self.device)
        src_lens = batch.get_src_lens()
        src_encodings, src_last_state, src_last_cell = self.encoder(src_tensor, src_lens)
        # omit the last word of tgt, which is </s>
        tgt_in_tensor = batch.get_tgt_in_tensor(self.tgt_vocab, self.device)
        # omit the first word of tgt, which is <s>
        tgt_out_tensor = batch.get_tgt_out_tensor(self.tgt_vocab, self.device)

        word_losses, ys = self.decoder(tgt_in_tensor, tgt_out_tensor, src_encodings, src_lens, src_last_state, src_last_cell)
        # (batch_size,)
        example_losses = word_losses.sum(dim=0)
        return example_losses

    def beam_search(self, example: Example, beam_size: int, max_dec_step: int,
                    BeamClass=Beam) -> List[Hypothesis]:
        batch = Batch([example])
        src_tensor = batch.get_src_tensor(self.src_vocab, self.device)
        src_lens = batch.get_src_lens()
        src_encodings, src_last_state, src_last_cell = self.encoder(src_tensor, src_lens)
        hypos = self.decoder.beam_search(example, beam_size, max_dec_step, BeamClass,
                                         src_encodings, src_lens, src_last_state, src_last_cell)
        return hypos

    def _get_sent_masks(self, max_len: int, sent_lens: List[int]):
        src_sent_masks = torch.zeros(len(sent_lens), max_len, dtype=FLOAT_TYPE)
        for e_id, l in enumerate(sent_lens):
            # make all paddings to 1
            src_sent_masks[e_id, l:] = 1
        return src_sent_masks.to(self.device)
