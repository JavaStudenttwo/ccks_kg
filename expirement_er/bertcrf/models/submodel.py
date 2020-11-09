import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from expirement_er.bertcrf.utils import torch_utils
from pytorchcrf import CRF
from expirement_er.bertcrf.models.mycrf import mycrf
import numpy as np

class SubjTypeModel(nn.Module):

    def __init__(self, opt, filter=3):
        super(SubjTypeModel, self).__init__()
        self.opt = opt
        self.num_tags = opt['num_tags']

        self.dropout = nn.Dropout(opt['dropout'])
        self.hidden_dim = opt['word_emb_dim']

        self.position_embedding = nn.Embedding(500, opt['position_emb_dim'])

        self.linear_subj = nn.Linear(self.hidden_dim, opt['num_tags'])
        self.init_weights()

        # 1.使用CRF工具包
        # self.crf_module = CRF(opt['num_tags'], False)

        # 2.使用mycrf
        if opt['decode_function'] == 'mycrf':
            self.crf_module = mycrf(opt['num_tags'])

        # 3.使用softmax
        if self.opt['decode_function'] == 'softmax':
            self.criterion = nn.CrossEntropyLoss(reduction='none')
            self.criterion.cuda()


    def init_weights(self):

        self.position_embedding.weight.data.uniform_(-1.0, 1.0)
        self.linear_subj.bias.data.fill_(0)
        init.xavier_uniform_(self.linear_subj.weight, gain=1)  # initialize linear layer

    def forward(self, hidden, tags, mask):

        subj_start_inputs = self.dropout(hidden)
        subj_start_logits = self.linear_subj(subj_start_inputs)

        # 1.使用CRF工具包
        # score = self.crf_module.forward(subj_start_logits, tags=tags, mask=mask)
        # return score

        # 2.使用mycrf
        if self.opt['decode_function'] == 'mycrf':
            Z = self.crf_module.forward(subj_start_logits, mask)
            score = self.crf_module.score(subj_start_logits, tags, mask)
            return torch.mean(Z - score)  # NLL loss

        # 3.使用softmax
        if self.opt['decode_function'] == 'softmax':
            # subj_start_logits = torch.softmax(subj_start_logits, dim=2)
            _s1 = subj_start_logits.view(-1, self.num_tags)
            tags = tags.view(-1).squeeze()
            loss = self.criterion(_s1, tags)
            loss = loss.view_as(mask)
            loss = torch.sum(loss.mul(mask.float())) / torch.sum(mask.float())
            return loss

    def predict_subj_start(self, hidden, mask):

        subj_start_logits = self.linear_subj(hidden)
        # 1.使用CRF工具包
        # best_tags_list = self.crf_module.decode(subj_start_logits)
        # return best_tags_list

        # 2.使用mycrf
        if self.opt['decode_function'] == 'mycrf':
            return self.crf_module.decode(subj_start_logits, mask)

        # 3.使用softmax
        if self.opt['decode_function'] == 'softmax':
            # subj_start_logits = torch.softmax(subj_start_logits, dim=2)
            best_tags_list = subj_start_logits.cpu().detach().numpy()
            best_tags_list = np.argmax(best_tags_list, 2).tolist()
            return best_tags_list
