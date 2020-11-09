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
from utils import torch_utils, loader
from models import layers, submodel


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
            inputs = [Variable(torch.LongTensor(b).cuda()) for b in batch[:3]]
            subj_start_binary = Variable(torch.LongTensor(batch[5]).cuda()).float()
            subj_end_binary = Variable(torch.LongTensor(batch[6]).cuda()).float()
            obj_start_relation = Variable(torch.LongTensor(batch[7]).cuda())
            obj_end_relation = Variable(torch.LongTensor(batch[8]).cuda())
            subj_start_type = Variable(torch.LongTensor(batch[9]).cuda())
            subj_end_type = Variable(torch.LongTensor(batch[10]).cuda())
            obj_start_type = Variable(torch.LongTensor(batch[11]).cuda())
            obj_end_type = Variable(torch.LongTensor(batch[12]).cuda())
            nearest_subj_start_position_for_each_token = Variable(torch.LongTensor(batch[13]).cuda())
            distance_to_nearest_subj_start = Variable(torch.LongTensor(batch[14]).cuda())
            distance_to_subj = Variable(torch.LongTensor(batch[15]).cuda())
            nearest_obj_start_position_for_each_token = Variable(torch.LongTensor(batch[3]).cuda())
            distance_to_nearest_obj_start = Variable(torch.LongTensor(batch[4]).cuda())
        else:
            inputs = [Variable(torch.LongTensor(b)) for b in batch[:4]]
            subj_start_label = Variable(torch.LongTensor(batch[4])).float()
            subj_end_label = Variable(torch.LongTensor(batch[5])).float()
            obj_start_label = Variable(torch.LongTensor(batch[6]))
            obj_end_label = Variable(torch.LongTensor(batch[7]))
            subj_type_start_label = Variable(torch.LongTensor(batch[8]))
            subj_type_end_label = Variable(torch.LongTensor(batch[9]))
            obj_type_start_label = Variable(torch.LongTensor(batch[10]))
            obj_type_end_label = Variable(torch.LongTensor(batch[11]))
            subj_nearest_start_for_each = Variable(torch.LongTensor(batch[12]))
            subj_distance_to_start = Variable(torch.LongTensor(batch[13]))
        
        
        mask = (inputs[0].data>0).float()
        # step forward
        self.model.train()
        self.optimizer.zero_grad()

        
        subj_start_logits, subj_end_logits, obj_start_logits, obj_end_logits = self.model(inputs, distance_to_subj)

        subj_start_loss = self.obj_criterion(subj_start_logits.view(-1, self.opt['num_subj_type']+1), subj_start_type.view(-1).squeeze()).view_as(mask)
        subj_start_loss = torch.sum(subj_start_loss.mul(mask.float()))/torch.sum(mask.float())
        
        subj_end_loss = self.obj_criterion(subj_end_logits.view(-1, self.opt['num_subj_type']+1), subj_end_type.view(-1).squeeze()).view_as(mask)
        subj_end_loss = torch.sum(subj_end_loss.mul(mask.float()))/torch.sum(mask.float())
        
        obj_start_loss = self.obj_criterion(obj_start_logits.view(-1, self.opt['num_class']+1), obj_start_relation.view(-1).squeeze()).view_as(mask)
        obj_start_loss = torch.sum(obj_start_loss.mul(mask.float()))/torch.sum(mask.float())
        
        obj_end_loss = self.obj_criterion(obj_end_logits.view(-1, self.opt['num_class']+1), obj_end_relation.view(-1).squeeze()).view_as(mask)
        obj_end_loss = torch.sum(obj_end_loss.mul(mask.float()))/torch.sum(mask.float())
        
        loss = self.opt['subj_loss_weight']*(subj_start_loss + subj_end_loss) + (obj_start_loss + obj_end_loss)
        
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

    def predict_obj_per_instance(self, inputs, hidden):

        if self.opt['cuda']:
            inputs = [Variable(torch.LongTensor(b).cuda()) for b in inputs]
        else:
            inputs = [Variable(torch.LongTensor(b)).unsqueeze(0) for b in inputs[:4]]
        mask = (inputs[0].data>0).float()

        words, subj_start_position, subj_end_position, distance_to_subj = inputs # unpack

        self.model.eval()

        obj_start_logits = self.model.obj_sublayer.predict_obj_start(hidden, subj_start_position, subj_end_position, distance_to_subj)
        obj_end_logits = self.model.obj_sublayer.predict_obj_end(hidden, subj_start_position, subj_end_position, distance_to_subj)

        return obj_start_logits, obj_end_logits


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
        self.obj_sublayer = submodel.ObjBaseModel(opt)
        self.opt = opt
        self.bert_model = bert_model
        self.topn = self.opt.get('topn', 1e10)
        self.use_cuda = opt['cuda']

    def based_encoder(self, words):
        hidden, _ = self.bert_model(words,output_all_encoded_layers=False)
        return hidden
    
    def forward(self, inputs, distance_to_subj):

        words, subj_start_position, subj_end_position = inputs

        hidden = self.based_encoder(words)
        subj_start_logits, subj_end_logits = self.subj_sublayer(hidden)
        obj_start_logits, obj_end_logits = self.obj_sublayer(hidden, subj_start_position, subj_end_position, distance_to_subj)

        return subj_start_logits, subj_end_logits, obj_start_logits, obj_end_logits
    

