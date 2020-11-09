import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from expirement_er.etlspan.utils import torch_utils


class SubjTypeModel(nn.Module):

    
    def __init__(self, opt, filter=3):
        super(SubjTypeModel, self).__init__()

        self.dropout = nn.Dropout(opt['dropout'])
        self.hidden_dim = opt['word_emb_dim']

        self.position_embedding = nn.Embedding(500, opt['position_emb_dim'])

        self.linear_subj_start = nn.Linear(self.hidden_dim, opt['num_subj_type']+1)
        self.linear_subj_end = nn.Linear(self.hidden_dim, opt['num_subj_type']+1)
        self.init_weights()



    def init_weights(self):

        self.position_embedding.weight.data.uniform_(-1.0, 1.0)
        self.linear_subj_start.bias.data.fill_(0)
        init.xavier_uniform_(self.linear_subj_start.weight, gain=1) # initialize linear layer

        self.linear_subj_end.bias.data.fill_(0)
        init.xavier_uniform_(self.linear_subj_end.weight, gain=1) # initialize linear layer

    def forward(self, hidden):

        subj_start_inputs = self.dropout(hidden)
        subj_start_logits = self.linear_subj_start(subj_start_inputs)

        subj_end_inputs = self.dropout(hidden)
        subj_end_logits = self.linear_subj_end(subj_end_inputs)

        return subj_start_logits.squeeze(-1), subj_end_logits.squeeze(-1)


    def predict_subj_start(self, hidden):

        subj_start_logits = self.linear_subj_start(hidden)
        subj_start_logits = torch.argmax(subj_start_logits, 2)

        return subj_start_logits.squeeze(-1)[0].data.cpu().numpy()

    def predict_subj_end(self, hidden):

        subj_end_logits = self.linear_subj_end(hidden)
        subj_end_logits = torch.argmax(subj_end_logits, 2)

        return subj_end_logits.squeeze(-1)[0].data.cpu().numpy()


