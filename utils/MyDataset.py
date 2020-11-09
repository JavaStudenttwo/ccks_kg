# # 定义dataset 以及 dataloader

# In[ ]:

import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
from utils.schemas import *
from torch.autograd import Variable


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, preprocessed_datas, tokenizer: BertTokenizer, max_length=256):
        self.preprocessed_datas = preprocessed_datas
        self.tokenizer = tokenizer
        self.max_length = max_length

    def pad_sent_ids(self, sent_ids, ts1, ts2, max_length, padded_token_id):
        mask = [1] * (min(len(sent_ids), max_length)) + [0] * (max_length - len(sent_ids))
        sent_ids = sent_ids[:max_length] + [padded_token_id] * (max_length - len(sent_ids))
        ts1 = ts1[:max_length] + [padded_token_id] * (max_length - len(ts1))
        ts2 = ts2[:max_length] + [padded_token_id] * (max_length - len(ts2))
        return sent_ids, ts1, ts2, mask

    def process_one_preprocessed_data(self, preprocessed_data):
        import copy
        preprocessed_data = copy.deepcopy(preprocessed_data)

        sent_token_ids = preprocessed_data['sent_token_ids']
        ts1 = preprocessed_data['ts1']
        ts2 = preprocessed_data['ts2']

        sent_token_ids, ts1, ts2, mask = self.pad_sent_ids(sent_token_ids, ts1, ts2, max_length = self.max_length, padded_token_id=0)

        T = np.array(sent_token_ids)
        S1 = np.array(ts1)
        S2 = np.array(ts2)
        mask = np.array(mask)

        preprocessed_data['T'] = Variable(torch.LongTensor(T).cuda())
        preprocessed_data['S1'] = Variable(torch.LongTensor(S1).cuda())
        preprocessed_data['S2'] = Variable(torch.LongTensor(S2).cuda())
        preprocessed_data['mask'] = Variable(torch.LongTensor(mask).cuda())

        return preprocessed_data

    def __getitem__(self, item):
        return self.process_one_preprocessed_data(
            self.preprocessed_datas[item]
        )

    def __len__(self):
        return len(self.preprocessed_datas)
