#!/usr/bin/env python
#-*- coding:utf-8 -*-
# file name: train.py
# file description:
# created at:5/10/2021 3:34 PM


"""
Usage:
  train.py [options] TRAIN_SET_SRC TRAIN_SET_TGT VALID_SET_SRC VALID_SET_TGT VOCAB_FILE

Options:
    -h --help                 show this screen.
    --cuda INT                use GPU [default: true]
    --src-max-len INT         max length of src [default: 100]
    --tgt-max-len INT         max length of tgt [default: 50]
    --model-class STR         model class [default: modules.Seq2Seq.Seq2Seq]
    --embed-size INT          embed size [default: 256]
    --enc-hidden-size INT     encoder hidden size [default: 512]
    --dec-hidden-size INT     hidden size [default: 512]
    --num-layers INT          number of layers [default: 1]
    --input-feed BOOL         use input feeding [default: true]
    --attn-type STR           choose an attention type, e.g., dot, general, and mlp. [default: mlp]
    --attn-func STR           choose an attention function, e.g., softmax and sparsemax. [default: softmax]
    --seed INT                random seed [default: 0]
    --uniform-init FLOAT      uniform initialization of parameters [default: 0.1]
    --train-batch-size INT    train batch size [default: 32]
    --valid-batch-size INT    valid batch size [default: 32]
    --lr FLOAT                learning rate [default: 0.001]
    --dropout-rate FLOAT      dropout rate [default: 0.2]
    --teacher-forcing FLOAT   teacher forcing ratio [default: 1.0]
    --clip-grad FLOAT         gradient clipping [default: 5.0]
    --log-every INT           log interval [default: 100]
    --valid-niter INT         validate interval [default: 500]
    --patience INT            wait for how many validations to decay learning rate [default: 5]
    --max-trial-num INT       terminal training after how many trials [default: 5]
    --lr-decay FLOAT          learning rate decay [default: 0.5]
    --max-epoch INT           max epoch [default: 50]
    --log-dir DIR             dir for tensorboard log [default: log/]
    --save-to FILE            model save path [default: model.bin]
    --example-class STR       Example Class used to load an example [default: dataset.Example]
"""

# Reference 1: https://github.com/pcyin/pytorch_basic_nmt
# Reference 2: https://github.com/Tbabm/CUP
import time
from abc import ABC, abstractmethod
from docopt import docopt
import logging
from modules.utils.common import *
import tensorflow as tf
from dataset import Dataset, Batch


class TFLogger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def scalar_dict_summary(self, info, step):
        for tag, value in info.items():
            self.scalar_summary(tag, value, step)


class LossReporter(object):
    def __init__(self, tf_logger: TFLogger = None):
        self._report_loss = 0
        self._cum_loss = 0
        self._report_tgt_words = 0
        self._cum_tgt_words = 0
        self._report_examples = 0
        self._cum_examples = 0
        self._train_begin_time = self._begin_time = time.time()
        self.tf_logger = tf_logger

    @property
    def report_tgt_words(self):
        return self._report_tgt_words

    @property
    def avg_loss_per_example(self):
        return self._report_loss / self._report_examples

    @property
    def avg_ppl(self):
        return np.exp(self._report_loss / self._report_tgt_words)

    @property
    def avg_cum_loss_per_example(self):
        return self._cum_loss / self._cum_examples

    @property
    def avg_cum_ppl(self):
        return np.exp(self._cum_loss / self._cum_tgt_words)

    def update(self, batch_loss, tgt_words_num, batch_size):
        self._report_loss += batch_loss
        self._cum_loss += batch_loss
        self._report_tgt_words += tgt_words_num
        self._cum_tgt_words += tgt_words_num
        self._report_examples += batch_size
        self._cum_examples += batch_size

    def reset_report_stat(self):
        self._report_loss = 0
        self._report_tgt_words = 0
        self._report_examples = 0
        self._train_begin_time = time.time()

    def reset_cum_stat(self):
        self._cum_loss = 0
        self._cum_examples = 0
        self._cum_tgt_words = 0

    def report(self, epoch, iter):
        train_time = time.time() - self._train_begin_time
        spend_time = time.time() - self._begin_time
        logging.info('epoch %d, iter %d, avg. loss %.6f, avg. ppl %.6f ' \
                     'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec'
                     % (epoch, iter, self.avg_loss_per_example, self.avg_ppl,
                        self._cum_examples, self.report_tgt_words / train_time, spend_time))
        if self.tf_logger:
            tf_info = {
                'train_loss': self.avg_loss_per_example,
                'train_ppl': self.avg_ppl,
            }
            self.tf_logger.scalar_dict_summary(tf_info, iter)

    def report_cum(self, epoch, iter):
        logging.info('epoch %d, iter %d, cum. loss %.6f, cum. ppl %.6f cum. examples %d'
                     % (epoch, iter, self.avg_cum_loss_per_example, self.avg_cum_ppl, self._cum_examples))
        if self.tf_logger:
            tf_info = {
                'cum_loss': self.avg_cum_loss_per_example,
                'cum_ppl': self.avg_cum_ppl
            }
            self.tf_logger.scalar_dict_summary(tf_info, iter)

    def report_valid(self, iter, ppl):
        logging.info('validation: iter %d, dev. ppl %f' % (iter, ppl))
        if self.tf_logger:
            self.tf_logger.scalar_summary("ppl", ppl, iter)


class Procedure(ABC):
    def __init__(self, args: dict):
        self._args = args
        self._model = None

    def _set_device(self):
        self._device = torch.device("cuda:0" if bool(self._args['--cuda']) else "cpu")
        logging.info("use device: {}".format(self._device))
        self._model.to(self._device)

    @abstractmethod
    def _init_model(self):
        pass


class Trainer(Procedure):
    def __init__(self, args: dict, tf_log: bool = True):
        super(Trainer, self).__init__(args)
        self._device = None
        self._cur_patience = 0
        # self._cur_trail = 0
        self._hist_valid_scores = []
        self.tf_logger = TFLogger(self._args['--log-dir']) if tf_log else None

    @property
    def _train_batch_size(self):
        return int(self._args['--train-batch-size'])

    @property
    def _valid_batch_size(self):
        return int(self._args['--valid-batch-size'])

    @property
    def _clip_grad(self):
        return float(self._args['--clip-grad'])

    @property
    def _log_every(self):
        return int(self._args['--log-every'])

    @property
    def _valid_niter(self):
        return int(self._args['--valid-niter'])

    @property
    def _model_save_path(self):
        return self._args['--save-to']

    @property
    def _max_patience(self):
        return int(self._args['--patience'])

    @property
    def _max_trial_num(self):
        return int(self._args['--max-trial-num'])

    @property
    def _max_epoch(self):
        return int(self._args['--max-epoch'])

    @property
    def _optim_save_path(self):
        return self._model_save_path + '.optim'

    def _uniform_init_model_params(self):
        uniform_init = float(self._args['--uniform-init'])
        if np.abs(uniform_init) > 0.:
            logging.info('uniformly initialize parameters [-{}, +{}]'.format(uniform_init, uniform_init))
            for name, p in self._model.named_parameters():
                p.data.uniform_(-uniform_init, uniform_init)

    def _init_optimizer(self):
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=float(self._args['--lr']))

    def train_a_batch(self, batch: Batch) -> float:
        self._optimizer.zero_grad()
        # (batch_size)
        example_losses = self._model(batch)
        batch_loss = example_losses.sum()
        loss = batch_loss / len(batch)
        loss.backward()
        # clip gradient
        if self._clip_grad != -1:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_grad)
        self._optimizer.step()
        return batch_loss.item()

    def save_model(self):
        logging.info('save currently the best model to [%s]' % self._model_save_path)
        self._model.save(self._model_save_path, self._args)
        # also save the optimizers' state
        torch.save(self._optimizer.state_dict(), self._optim_save_path)

    def load_model(self):
        logging.info('load previously best model')
        params = torch.load(self._model_save_path, map_location=lambda storage, loc: storage)
        self._model.load_state_dict(params['state_dict'])
        self._model.to(self._device)

        logging.info('restore parameters of the optimizers')
        self._optimizer.load_state_dict(torch.load(self._optim_save_path))

    def decay_lr(self):
        # decay lr, and restore from previously best checkpoint
        lr = self._optimizer.param_groups[0]['lr'] * float(self._args['--lr-decay'])
        logging.info('decay learning rate to %f' % lr)
        self.load_model()

        # set new lr
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def _validate(self, val_set):
        was_training = self._model.training
        self._model.eval()
        cum_loss = 0
        cum_tgt_words = 0
        with torch.no_grad():
            for batch in val_set.train_batch_iter(self._valid_batch_size, shuffle=False):
                batch_loss = self._model(batch).sum()
                cum_loss += batch_loss.item()
                cum_tgt_words += batch.tgt_words_num
            dev_ppl = np.exp(cum_loss / cum_tgt_words)
        # negative: the larger the better
        valid_metric = -dev_ppl

        if was_training:
            self._model.train()

        return valid_metric

    def validate(self, train_iter, val_set, loss_reporter):
        logging.info('begin validation ...')

        valid_metric = self._validate(val_set)
        loss_reporter.report_valid(train_iter, valid_metric)

        is_better = len(self._hist_valid_scores) == 0 or valid_metric > max(self._hist_valid_scores)
        self._hist_valid_scores.append(valid_metric)

        return is_better

    def _init_model(self):
        model_class = get_attr_by_name(self._args['--model-class'])
        self._model = model_class(*model_class.prepare_model_params(self._args))
        self._model.train()

        self._uniform_init_model_params()

        self._set_device()
        self._init_optimizer()

    def load_dataset(self):
        logging.info("Load example using {}".format(self._args['--example-class']))
        example_class = get_attr_by_name(self._args['--example-class'])
        train_set = Dataset.create_from_file(self._args['TRAIN_SET_SRC'], self._args['TRAIN_SET_TGT'],
                                             self._args['--src-max-len'], self._args['--tgt-max-len'], example_class)
        val_set = Dataset.create_from_file(self._args['VALID_SET_SRC'], self._args['VALID_SET_TGT'],
                                           self._args['--src-max-len'], self._args['--tgt-max-len'], example_class)
        return train_set, val_set

    def train(self):
        train_set, val_set = self.load_dataset()
        self._init_model()

        epoch = train_iter = 0
        loss_reporter = LossReporter(self.tf_logger)
        logging.info("Start training")
        while True:
            epoch += 1
            for batch in train_set.train_batch_iter(batch_size=self._train_batch_size, shuffle=True):
                train_iter += 1
                batch_loss_val = self.train_a_batch(batch)
                # tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
                loss_reporter.update(batch_loss_val, batch.tgt_words_num, len(batch))

                if train_iter % self._log_every == 0:
                    loss_reporter.report(epoch, train_iter)
                    loss_reporter.reset_report_stat()

                if train_iter % self._valid_niter == 0:
                    loss_reporter.report_cum(epoch, train_iter)
                    loss_reporter.reset_cum_stat()

                    is_better = self.validate(train_iter, val_set, loss_reporter)
                    if is_better:
                        self._cur_patience = 0
                        self.save_model()
                    else:
                        self._cur_patience += 1
                        logging.info('hit patience {}'.format(self._cur_patience))
                        if self._cur_patience == self._max_patience:
                            # self._cur_trail += 1
                            # logging.info('hit #{} trial'.format(self._cur_trail))
                            # if self._cur_trail == self._max_trial_num:
                            logging.info('early stop!')
                            return
                        self.decay_lr()
                            # # reset patience
                            # self._cur_patience = 0
            if epoch == self._max_epoch:
                logging.info('reached maximum number of epochs')
                return


def train(args):
    logging.debug("Train with args:")
    logging.info(args)

    # set reproducibility
    if args['--seed'] is not None:
        seed = int(args['--seed'])
        set_reproducibility(seed)

    trainer = Trainer(args)
    trainer.train()


def main():
    args = docopt(__doc__)
    print(args)
    train(args)


if __name__ == '__main__':
    main()

