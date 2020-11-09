# -*- coding: utf-8 -*-

from .BasicModule import BasicModule
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from expirement_re.bert_one_mis import utils


class bert_one_mis(BasicModule):

    def __init__(self, opt):
        super(bert_one_mis, self).__init__()

        self.opt = opt

        self.model_name = 'bert_one_mis'

        self.bert_model = BertModel.from_pretrained(opt.bert_path)
        self.bert_model.cuda()

        hidden_dim = self.opt.hidden_dim
        rel_num = self.opt.rel_num

        self.linear = nn.Linear(hidden_dim, rel_num)
        self.dropout = nn.Dropout(self.opt.drop_out)


    def forward(self, x, train=False):

        ent1, ent2, select_num, select_lab, select_sen = x

        hidden, _ = self.bert_model(select_sen, output_all_encoded_layers=False)
        out = self.dropout(_)
        out = self.linear(out)
        # out = torch.argmax(out, dim=1)

        return out
