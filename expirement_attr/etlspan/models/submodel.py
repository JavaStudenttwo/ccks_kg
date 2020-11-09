import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from utils import torch_utils

from models.layers import *



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

        return subj_start_logits.squeeze(-1)[0].data.cpu().numpy()

    def predict_subj_end(self, hidden):

        subj_end_logits = self.linear_subj_end(hidden)

        return subj_end_logits.squeeze(-1)[0].data.cpu().numpy()


class ObjBaseModel(nn.Module):

    def __init__(self, opt, filter=3):
        super(ObjBaseModel, self).__init__()

        self.dropout = self.drop = nn.Dropout(opt['dropout'])
        self.distance_to_subj_embedding = nn.Embedding(400, opt['position_emb_dim'])
        self.distance_to_obj_start_embedding = nn.Embedding(500, opt['position_emb_dim'])
        self.input_dim = opt['obj_input_dim']

        self.linear_obj_start = nn.Linear(self.input_dim, opt['num_class']+1)
        self.linear_obj_end = nn.Linear(self.input_dim, opt['num_class']+1)
        self.init_weights()

    def init_weights(self):
        self.linear_obj_start.bias.data.fill_(0)
        init.xavier_uniform_(self.linear_obj_start.weight, gain=1) # initialize linear layer
        self.linear_obj_end.bias.data.fill_(0)
        init.xavier_uniform_(self.linear_obj_end.weight, gain=1) # initialize linear layer
        self.distance_to_subj_embedding.weight.data.uniform_(-1.0, 1.0)
        self.distance_to_obj_start_embedding.weight.data.uniform_(-1.0, 1.0)

    def forward(self, hidden, subj_start_position, subj_end_position, distance_to_subj):


        batch_size, seq_len, input_size = hidden.shape

        subj_start_hidden = torch.gather(hidden, dim=1, index=subj_start_position.unsqueeze(2).repeat(1,1,input_size)).squeeze(1)       
        subj_end_hidden = torch.gather(hidden, dim=1, index=subj_end_position.unsqueeze(2).repeat(1,1,input_size)).squeeze(1)       
        distance_to_subj_emb = self.distance_to_subj_embedding(distance_to_subj+200)   # To avoid negative indices     

        subj_related_info = torch.cat([seq_and_vec(seq_len,subj_start_hidden), seq_and_vec(seq_len,subj_end_hidden), distance_to_subj_emb], dim=2)
        obj_inputs = torch.cat([hidden, subj_related_info], dim=2)    
        obj_inputs = self.dropout(obj_inputs)

        obj_start_outputs = self.dropout(obj_inputs)
        obj_start_logits = self.linear_obj_start(obj_start_outputs)

        obj_end_outputs = self.dropout(obj_inputs)
        obj_end_logits = self.linear_obj_end(obj_end_outputs)

        return obj_start_logits, obj_end_logits

    def predict_obj_start(self, hidden, subj_start_position, subj_end_position, distance_to_subj):

        batch_size, seq_len, input_size = hidden.size()

        subj_start_hidden = torch.gather(hidden, dim=1, index=subj_start_position.unsqueeze(2).repeat(1,1,input_size)).squeeze(1)       
        subj_end_hidden = torch.gather(hidden, dim=1, index=subj_end_position.unsqueeze(2).repeat(1,1,input_size)).squeeze(1)       
        distance_to_subj_emb = self.distance_to_subj_embedding(distance_to_subj+200)        
        subj_related_info = torch.cat([seq_and_vec(seq_len,subj_start_hidden), seq_and_vec(seq_len,subj_end_hidden), distance_to_subj_emb], dim=2)
        obj_inputs = torch.cat([hidden, subj_related_info], dim=2)
        obj_inputs = self.dropout(obj_inputs)

        obj_start_logits = self.linear_obj_start(obj_inputs)

        return obj_start_logits.squeeze(-1)[0].data.cpu().numpy()



    def predict_obj_end(self, hidden, subj_start_position, subj_end_position, distance_to_subj):

        batch_size, seq_len, input_size = hidden.size()

        subj_start_hidden = torch.gather(hidden, dim=1, index=subj_start_position.unsqueeze(2).repeat(1, 1, input_size)).squeeze(1)
        subj_end_hidden = torch.gather(hidden, dim=1, index=subj_end_position.unsqueeze(2).repeat(1, 1, input_size)).squeeze(1)
        distance_to_subj_emb = self.distance_to_subj_embedding(distance_to_subj + 200)
        subj_related_info = torch.cat([seq_and_vec(seq_len, subj_start_hidden), seq_and_vec(seq_len, subj_end_hidden), distance_to_subj_emb],dim=2)
        obj_inputs = torch.cat([hidden, subj_related_info], dim=2)
        obj_inputs = self.dropout(obj_inputs)

        obj_end_logits = self.linear_obj_end(obj_inputs)

        return obj_end_logits.squeeze(-1)[0].data.cpu().numpy()
