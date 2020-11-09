"""
A joint model for relation extraction, written in pytorch.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
from expirement_er.etlspan.utils import torch_utils, loader
from expirement_er.etlspan.models import submodel


class RelationModel(object):
    """ A wrapper class for the training and evaluation of models. """
    def __init__(self, opt, bert_model):
        self.opt = opt
        self.bert_model = bert_model
        self.model = BiLSTMCNN(opt, bert_model)
        self.subj_criterion = nn.BCELoss(reduction='none')
        self.obj_criterion = nn.CrossEntropyLoss(reduction='none')
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.subj_criterion.cuda()
            self.obj_criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'], opt['weight_decay'])
    

    def update(self, batch):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            inputs = Variable(torch.LongTensor(batch[0]).cuda())
            subj_start_type = Variable(torch.LongTensor(batch[3]).cuda())
            subj_end_type = Variable(torch.LongTensor(batch[4]).cuda())

        mask = (inputs.data>0).float()
        self.model.train()
        self.optimizer.zero_grad()

        subj_start_logits, subj_end_logits = self.model(inputs)
        subj_start_logits = subj_start_logits.view(-1, self.opt['num_subj_type']+1)
        subj_start_type = subj_start_type.view(-1).squeeze()
        subj_start_loss = self.obj_criterion(subj_start_logits, subj_start_type).view_as(mask)
        subj_start_loss = torch.sum(subj_start_loss.mul(mask.float())) / torch.sum(mask.float())

        subj_end_loss = self.obj_criterion(subj_end_logits.view(-1, self.opt['num_subj_type']+1), subj_end_type.view(-1).squeeze()).view_as(mask)
        subj_end_loss = torch.sum(subj_end_loss.mul(mask.float())) / torch.sum(mask.float())

        loss = subj_start_loss + subj_end_loss
        
        # backward
        loss.backward()
        # torch.nn.utils.clip_grad_norm(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val


    def predict_subj_per_instance(self, words):

        if self.opt['cuda']:
            words = Variable(torch.LongTensor(words).cuda())
            # chars = Variable(torch.LongTensor(chars).cuda())
            # pos_tags = Variable(torch.LongTensor(pos_tags).cuda())
        else:
            words = Variable(torch.LongTensor(words))
            # features = Variable(torch.LongTensor(features))

        batch_size, seq_len = words.size()
        mask = (words.data>0).float()
        # forward
        self.model.eval()
        hidden = self.model.based_encoder(words)

        subj_start_logits = self.model.subj_sublayer.predict_subj_start(hidden)
        subj_end_logits = self.model.subj_sublayer.predict_subj_end(hidden)
        
        return subj_start_logits, subj_end_logits, hidden


    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                'epoch': epoch
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

class BiLSTMCNN(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, bert_model):
        super(BiLSTMCNN, self).__init__()
        self.drop = nn.Dropout(opt['dropout'])
        self.input_size = opt['word_emb_dim']
        self.subj_sublayer = submodel.SubjTypeModel(opt)
        self.opt = opt
        self.bert_model = bert_model
        self.topn = self.opt.get('topn', 1e10)
        self.use_cuda = opt['cuda']

    def based_encoder(self, words):
        hidden, _ = self.bert_model(words, output_all_encoded_layers=False)
        return hidden
    
    def forward(self, inputs):

        hidden = self.based_encoder(inputs)
        subj_start_logits, subj_end_logits = self.subj_sublayer(hidden)

        return subj_start_logits, subj_end_logits
    

