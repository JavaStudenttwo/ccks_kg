# -*- coding: utf-8 -*-

from .BasicModule import BasicModule
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

class bert_att_mis(BasicModule):

    def __init__(self, opt):
        super(bert_att_mis, self).__init__()

        self.opt = opt
        self.model_name = 'bert_att_mis'
        self.test_scale_p = 0.5

        self.bert_model = BertModel.from_pretrained(opt.bert_path)
        self.bert_model.cuda()

        self.bags_feature = []

        rel_dim = opt.rel_dim

        self.rel_embs = nn.Parameter(torch.randn(self.opt.rel_num, rel_dim))
        self.rel_bias = nn.Parameter(torch.randn(self.opt.rel_num))

        self.dropout = nn.Dropout(self.opt.drop_out)

        self.init_model_weight()

    def init_model_weight(self):
        '''
        use xavier to init
        '''
        nn.init.xavier_uniform(self.rel_embs)
        nn.init.uniform(self.rel_bias)

    def forward(self, x, label=None):
        # get all sentences embedding in all bags of one batch
        self.bags_feature = self.get_bags_feature(x)

        if label is None:
            return self.test(x)
        else:
            return self.fit(label)

    def fit(self, label):
        '''
        train process
        '''
        x = self.get_batch_feature(label)               # batch_size * sentence_feature_num
        x = self.dropout(x)
        out = x.mm(self.rel_embs.t()) + self.rel_bias     # o = Ms + d (formual 10 in paper)

        if self.opt.use_gpu:
            v_label = torch.LongTensor(label).cuda()
        else:
            v_label = torch.LongTensor(label)
        ce_loss = F.cross_entropy(out, Variable(v_label))
        return ce_loss

    def test(self, x):
        '''
        test process
        '''
        pre_y = []
        for label in range(0, self.opt.rel_num):
            labels = [label for _ in range(len(x))]                 # generate the batch labels
            bags_feature = self.get_batch_feature(labels)
            out = self.test_scale_p * bags_feature.mm(self.rel_embs.t()) + self.rel_bias
            # out = F.softmax(out, 1)
            # pre_y.append(out[:, label])
            pre_y.append(out.unsqueeze(1))

        # return pre_y
        res = torch.cat(pre_y, 1).max(1)[0]
        return torch.argmax(F.softmax(res, 1), 1)

    def get_batch_feature(self, labels):
        '''
        Using Attention to get all bags embedding in a batch
        '''
        batch_feature = []

        for bag_embs, label in zip(self.bags_feature, labels):
            # calculate the weight: xAr or xr
            alpha = bag_embs.mm(self.rel_embs[label].view(-1, 1))
            # alpha = bag_embs.mm(self.att_w[label]).mm(self.rel_embs[label].view(-1, 1))
            bag_embs = bag_embs * F.softmax(alpha, 0)
            bag_vec = torch.sum(bag_embs, 0)
            batch_feature.append(bag_vec.unsqueeze(0))

        return torch.cat(batch_feature, 0)

    def get_bags_feature(self, bags):
        '''
        get all bags embedding in one batch before Attention
        '''
        bags_feature = []
        for bag in bags:
            if self.opt.use_gpu:
                data = map(lambda x: Variable(torch.LongTensor(x).cuda()), bag)
            else:
                data = map(lambda x: Variable(torch.LongTensor(x)), bag)

            ent1, ent2, bags_num, lab, sen = data
            hidden, out = self.bert_model(sen, output_all_encoded_layers=False)
            bags_feature.append(out)

        return bags_feature


