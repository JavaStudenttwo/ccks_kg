# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import os
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


class PCNNData(Dataset):

    def __init__(self, opt, train=True):
        self.opt = opt
        self.bert_tokenizer = BertTokenizer.from_pretrained(opt.bert_tokenizer_path)
        if train:
            path = os.path.join(opt.root_path, 'train_mulit.txt')
            print('loading train data')
        else:
            path = os.path.join(opt.root_path, 'valid_mulit.txt')
            print('loading valid data')

        self.x = self.parse_sen(path)
        print('loading finish')

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)

    def parse_sen(self, path):
        all_bags = []

        str1 = list('实体')
        str2 = list('和实体')
        str3 = list('之间的关系是什么？')

        f = open(path, encoding='utf-8')
        while 1:

            question = []

            # if len(all_bags) > 200:
            #     break
            # 第一行读取实体和关系
            line = f.readline()
            if not line:
                break
            entities = list(map(str, line.split('-*-')))
            entitity = entities[:2]
            pre = entities[2][0:-1]
            label = self.opt.label2id[pre]
            # 第二行读取句子数量
            line = f.readline()
            sent_num = int(line)

            ent1_str = self.bert_tokenizer.tokenize(entitity[0].replace(' ', ''))
            ent1 = self.bert_tokenizer.convert_tokens_to_ids(ent1_str)
            ent2_str = self.bert_tokenizer.tokenize(entitity[1].replace(' ', ''))
            ent2 = self.bert_tokenizer.convert_tokens_to_ids(ent2_str)

            question = ['[CLS]'] + str1 + ent1_str + str2 + ent2_str + str3 + ['[SEP]']

            # 通过MRC问答的方式加入实体
            # 实体1 和 实体2 的关系是什么？
            sentences = []
            for i in range(0, sent_num):
                sent = f.readline()
                sent = sent.replace(' ', '')
                sent = self.bert_tokenizer.tokenize(sent)
                sent = question + sent + ['[SEP]']
                word = self.bert_tokenizer.convert_tokens_to_ids(sent[:])
                sentences.append(word)

            bag = [ent1, ent2, sent_num, label, sentences]
            all_bags += [bag]
        return all_bags



