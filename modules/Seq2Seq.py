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
        self.dec_state_init = None
        self.decoder = None

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

    def _get_sent_masks(self, max_len: int, sent_lens: List[int]):
        src_sent_masks = torch.zeros(len(sent_lens), max_len, dtype=FLOAT_TYPE)
        for e_id, l in enumerate(sent_lens):
            # make all paddings to 1
            src_sent_masks[e_id, l:] = 1
        return src_sent_masks.to(self.device)

    def construct_src_input(self, batch):
        src_tensor = batch.get_src_tensor(self.src_vocab, self.device)
        src_lens = batch.get_src_lens()
        return src_tensor, src_lens

    @property
    def encoder_output_names(self):
        return ["edit_encodings", "edit_last_state", "edit_sent_masks", "src_encodings", "src_sent_masks",
                "dec_init_state"]

    def prepare_tgt_out_tensor(self, batch: Batch) -> Tensor:
        return batch.get_tgt_out_tensor(self.nl_vocab, self.device)

    def prepare_decoder_kwargs(self, encoder_output: dict, batch: Batch) -> dict:
        pass

    def forward(self, batch: Batch) -> Tensor:
        input_tensor = self.construct_src_input(batch)
        encoder_output = self.encoder(*input_tensor)
        print(encoder_output)
        assert False, "stop"
        tgt_in_tensor = batch.get_tgt_in_tensor(self.nl_vocab, self.device)
        tgt_out_tensor = self.prepare_tgt_out_tensor(batch)


        decoder_kwargs = self.prepare_decoder_kwargs(encoder_output, batch)
        # omit the last word of tgt, which is </s>
        # (tgt_sent_len - 1, batch_size, hidden_size)
        word_losses, ys = self.nl_decoder(tgt_in_tensor, tgt_out_tensor, **decoder_kwargs)

        # (batch_size,)
        example_losses = word_losses.sum(dim=0)
        return example_losses

    def beam_search(self, example: Example, beam_size: int, max_dec_step: int,
                    BeamClass=Beam) -> List[Hypothesis]:
        batch = Batch([example])
        input_tensor = self.construct_src_input(batch)
        encoder_output = self.encode(*input_tensor)
        # for idx, key in enumerate(encoder_output.keys()):
        #     print(key)
        #     print(encoder_output[key].size())
        #     if idx + 2 == len(encoder_output):
        #         break
        decoder_kwargs = self.prepare_decoder_kwargs(encoder_output, batch)
        hypos = self.nl_decoder.beam_search(example, beam_size, max_dec_step, BeamClass, **decoder_kwargs)
        return hypos
