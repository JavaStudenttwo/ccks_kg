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
from expirement_er.bertcrf.utils import torch_utils, loader
from expirement_er.bertcrf.models import submodel


class RelationModel(object):
    """ A wrapper class for the training and evaluation of models. """
    def __init__(self, opt, bert_model):
        self.opt = opt
        self.bert_model = bert_model
        self.model = BiLSTMCNN(opt, bert_model)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'], opt['weight_decay'])
    

    def update(self, batch):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            inputs = Variable(torch.LongTensor(batch[0]).cuda())
            tags = Variable(torch.LongTensor(batch[1]).cuda())

        mask = (inputs.data>0).float()
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.model(inputs, tags, mask)

        loss.backward()
        # torch.nn.utils.clip_grad_norm(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val


    def predict_subj_per_instance(self, words):

        if self.opt['cuda']:
            words = Variable(torch.LongTensor(words).cuda())
        mask = (words.data > 0).float()

        # forward
        self.model.eval()
        hidden = self.model.based_encoder(words)
        best_tags_list = self.model.subj_sublayer.predict_subj_start(hidden, mask)
        
        return best_tags_list


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
    
    def forward(self, inputs, tags, mask):

        hidden = self.based_encoder(inputs)
        loss = self.subj_sublayer(hidden, tags, mask)
        return loss
    

