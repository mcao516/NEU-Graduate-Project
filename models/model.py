#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from models.reg_model import REGModel
from models.data_utils import minibatches
from models.general_utils import Progbar
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, MultiStepLR
    

class REGShell(object):
    """Class for REG model training, evaluation and test."""

    def __init__(self, config, device, write_summary=True):
        super(REGShell, self).__init__()
        self.config = config
        self.logger = config.logger

        # set training device
        self.device = device

        self.logger.info("- Start building REG model...")
        self.regModel = self._build_model() # build model
        
        # multi-gpu training
        if torch.cuda.device_count() > 1 and config.multi_gpu:
            self.logger.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.regModel)
        self.regModel.to(device)
        
        self.optimizer = self._get_optimizer(self.regModel)
        self.scheduler = self._get_scheduler(self.optimizer)

        # create summary for tensorboard visualization
        if write_summary:
            self.writer = SummaryWriter(self.config.path_summary)
        else:
            self.writer = None
        
    def _build_model(self):
        """Build reference expression generation model.
        """
        return REGModel(self.config)

    def _get_optimizer(self, model):
        """Create optimizer for training.
        """
        return torch.optim.Adam(model.parameters(), 
                                lr=self.config.lr, 
                                weight_decay=self.config.l2_reg) 

    def _get_scheduler(self, optimizer):
        """Learning rate scheduler for Cosine Annealing.
        """
        if self.config.scheduler == 'exp':
            return ExponentialLR(optimizer, self.config.lr_decay)
        elif self.config.scheduler == 'multi':
            return MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1)
        elif self.config.scheduler == 'cosine':
            return CosineAnnealingLR(optimizer, 20, eta_min=0)
        else:
            raise ValueError("Unknown scheduler type!")

    def save_model(self):
        """Saves session = weights"""
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        
        save_path = self.config.dir_model + 'checkpoint.pth.tar'
        torch.save(self.regModel.state_dict(), save_path)
        self.logger.info("- model saved at: {}".format(save_path))
        
    def restore_model(self, dir_model):
        self.regModel.load_state_dict(torch.load(dir_model))
        self.logger.info("- model restored from: {}".format(dir_model))

    def run_epoch(self, train, dev, epoch, samples=None):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better
            
        """
        # train mode
        self.regModel.train()

        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        epoch_loss = 0.
        for i, batch in enumerate(minibatches(train, batch_size)):
            # convert numpy data into torch tensor
            context_input, des_input, dec_input = self.data_prepare(batch)
            total_losses = self.regModel(context_input, des_input, dec_input, decode_flag=False)
            batch_loss = total_losses.mean()
            batch_loss.backward() # backward propagation

            # gradient clipping by norm
            if self.config.grad_clip > 0:
                clip_grad_norm_(self.regModel.parameters(), self.config.grad_clip)
            
            # update
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # log loss information
            batch_loss = batch_loss.item()
            epoch_loss += batch_loss
            
            prog.update(i + 1, [("train loss",  batch_loss)])
            self.writer.add_scalar('batch_loss', batch_loss, epoch * nbatches + i + 1)
        
        epoch_loss = epoch_loss / nbatches
        self.scheduler.step() # update learning rate

        return epoch_loss


    def train(self, train, dev, sample_set=None):
        """Performs training with early stopping and lr exponential decay
        """
        self.config.log_info()
        self.logger.info('start training...')
        best_score, nepoch_no_imprv = 0, 0 # for early stopping
        for epoch in range(self.config.nepochs):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))

            # shuffle the dataset
            if self.config.shuffle_dataset: train.shuffle()
            epoch_loss = self.run_epoch(train, dev, epoch)

            # print out samples
            if sample_set is not None and self.config.print_samples:
                self.logger.info('Evaluating samples...')
                for pres, contexts, des, refex, oovs in zip(self.predict(sample_set)[:-1]):
                    self.displayOutput(pres, contexts, des, refex, oovs)

            # evaluate the model
            self.logger.info('Evaluating development set...')
            metrics = self.run_evaluate(dev)
            msg = " - ".join(["{} {:04.2f}".format(k, v)
                    for k, v in metrics.items()])
            self.logger.info(msg)
            score = metrics["acc"]

            # write summary
            self.writer.add_scalar('epoch_loss', epoch_loss, epoch + 1)
            self.writer.add_scalar('evaluate_acc', score, epoch + 1)
            self.writer.add_scalar('lr', self.scheduler.get_lr()[0], epoch + 1)

            # early stopping and saving best parameters
            if score >= best_score:
                nepoch_no_imprv = 0
                self.save_model()
                best_score = score
                self.logger.info("- new best score!")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    self.logger.info("- early stopping {} epochs without "\
                            "improvement".format(nepoch_no_imprv))
                    break

    def displayOutput(self, pres, context, description, refex, oovs, show_ground_truth=True):
        """Print out prediction samples during training.
        """
        if show_ground_truth:
            self.logger.info('- CONTEXT: {}'.format(context))
            self.logger.info('- DESCRIPTION: {}'.format(description))
            self.logger.info('- REFEX: {}'.format(refex))
        for i, pred in enumerate(pres):
            # [0, nwords - 1] [nwords, nwords+m-1]
            self.logger.info('- #{}: {}'.format((i+1), pred))

    def predict_batch(self, context_input, des_input, beam_search=True):
        """Predict referring expression on a batch of data

           Returns:
               preds: list of ids in greedy mode, list of list of ids in beam search mode
        """
        # self.regModel.eval() # set model to eval mode
        preds, switches = self.regModel(context_input, des_input, None, decode_flag=True, 
                              beam_search=beam_search)
        # self.regModel.train() # set model to train mode
        return preds, switches

    def evaluate(self, test):
        """Evaluate model on test set

        Args:
            test: instance of class Dataset
        """
        self.logger.info("Testing model over test set...")
        if self.config.beam_search:
            self.logger.info("- beam searching")
        else:
            self.logger.info("- greedy decoding")
        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)
    
    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields Example object

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...
            
        """
        self.regModel.eval() # set model to eval mode

        # progbar stuff for logging
        batch_size = self.config.batch_size_eva
        nbatches = (len(test) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        total, correct = 0., 0.
        for i, batch in enumerate(minibatches(test, 1)):
            context_input, des_input, _ = self.data_prepare(batch)
            preds, _ = self.predict_batch(context_input, des_input, 
                beam_search=self.config.beam_search)
            target = batch.target_batch[0].tolist() # [1, max_dec_len]
            if self.config.beam_search:
                pred = preds[0] # in beam search mode, find the sequence with the highest probability
            else:
                pred = preds 

            stop_id = self.config.stop_id
            if stop_id in pred and stop_id in target:
                pred_trunc = pred[1: pred.index(stop_id) + 1] # get rid of start token
                target_trunc = target[: target.index(stop_id) + 1]

                if pred_trunc == target_trunc:
                    correct += 1
            elif pred == target:
                correct += 1
            total += 1
            # update progress bar
            prog.update(i + 1, [("acc", correct/total)])
        acc = correct / total
        
        return {"acc": 100*acc}
    

    def predict(self, dataset):
        """Predict referring expression
        """
        self.regModel.eval() # set model to eval mode
        # progbar stuff for logging
        batch_size = self.config.batch_size_eva
        nbatches = (len(dataset) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)
        all_preds, all_contexts, all_des, all_refex, all_oovs = [], [], [], [], []
        all_switches = []
        if self.config.beam_search:
            self.logger.info("- beam searching")
        else:
            self.logger.info("- greedy decoding")

        for i, batch in enumerate(minibatches(dataset, 1)):
            context_input, des_input, _ = self.data_prepare(batch)
            preds, switches = self.predict_batch(context_input, des_input, beam_search=self.config.beam_search)
            # only select the most 3 possible beams
            if self.config.beam_search and len(preds) > 3:
                assert type(preds[0]) == list
                preds = preds[:3]
                
            all_switches.append(switches)
            all_preds.append(preds) # all_preds: [dataset_size, 3]
            all_contexts.append(batch.original_context[0]) # all_context: [dataset_size]
            all_des.append(batch.original_description[0])
            all_refex.append(batch.original_refex[0])
            all_oovs.append(batch.des_oovs[0])

            # update progress bar
            prog.update(i + 1)
        # convert predicted referring expression into strings
        pred_strings = []
        for i, (preds, oovs) in enumerate(zip(all_preds, all_oovs)):
            if type(preds[0]) == int:
                generated = ' '.join([self.config.id2word[ind] if ind < self.config.nwords else oovs[ind % self.config.nwords] for ind in preds])
                pred_strings.append([generated])
            else:
                beams = []
                for pred in preds:
                    generated = ' '.join([self.config.id2word[ind] if ind < self.config.nwords else oovs[ind % self.config.nwords] for ind in pred])
                    beams += [generated]
                pred_strings.append(beams)

        return pred_strings, all_contexts, all_des, all_refex, all_oovs, all_switches
    
    def data_prepare(self, batch):
        """Convert numpy data to tensor for training
        """
        # context input
        prec_word_ids = torch.tensor(batch.prec_word_ids, device=self.device, dtype=torch.long)
        prec_seq_lens = torch.tensor(batch.prec_seq_lens, device=self.device, dtype=torch.long)
        prec_char_ids = torch.tensor(batch.prec_char_ids, device=self.device, dtype=torch.long)
        prec_word_lens = torch.tensor(batch.prec_word_lens, device=self.device, dtype=torch.long)

        posc_word_ids = torch.tensor(batch.posc_word_ids, device=self.device, dtype=torch.long)
        posc_seq_lens = torch.tensor(batch.posc_seq_lens, device=self.device, dtype=torch.long)
        posc_char_ids = torch.tensor(batch.posc_char_ids, device=self.device, dtype=torch.long)
        posc_word_lens = torch.tensor(batch.posc_word_lens, device=self.device, dtype=torch.long)    

        _pre = prec_word_ids, prec_seq_lens, prec_char_ids, prec_word_lens
        _pos = posc_word_ids, posc_seq_lens, posc_char_ids, posc_word_lens
        context_input = _pre, _pos

        # description input
        des_word_ids = torch.tensor(batch.des_word_ids, device=self.device, dtype=torch.long)
        des_seq_lens = torch.tensor(batch.des_seq_lens, device=self.device, dtype=torch.long)
        des_mask = torch.tensor(batch.des_mask, device=self.device, dtype=torch.long)
        des_char_ids = torch.tensor(batch.des_char_ids, device=self.device, dtype=torch.long)
        des_word_lens = torch.tensor(batch.des_word_lens, device=self.device, dtype=torch.long)
        max_des_oovs = torch.tensor(batch.max_des_oovs, device=self.device, dtype=torch.long)
        des_extented_ids = torch.tensor(batch.des_extented_ids, device=self.device, dtype=torch.long)
        des_input = des_word_ids, des_seq_lens, des_mask, des_char_ids, \
            des_word_lens, max_des_oovs, des_extented_ids
        
        # decoding input
        _input = torch.tensor(batch.dec_batch, device=self.device, dtype=torch.long)
        target = torch.tensor(batch.target_batch, device=self.device, dtype=torch.long)
        dec_lens = torch.tensor(batch.dec_lens, device=self.device, dtype=torch.long)
        dec_mask = torch.tensor(batch.dec_padding_mask, device=self.device, dtype=torch.long)
        dec_input = _input, target, dec_lens, dec_mask

        return context_input, des_input, dec_input
    
