"""
A joint model for relation extraction, written in pytorch.
"""
import torch
from torch import nn
from torch.nn import init
from utils import torch_utils


class EntityModel(object):
    """ A wrapper class for the training and evaluation of models. """

    def __init__(self, opt, bert_model):
        self.opt = opt
        self.obj_criterion = nn.CrossEntropyLoss(reduction='none')
        self.model = BiLSTMCNN(opt, bert_model)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.obj_criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'], opt['weight_decay'])

    def update(self, T, S1, S2, mask):

        inputs = T
        subj_start_type = S1
        subj_end_type = S2

        self.model.train()
        self.optimizer.zero_grad()

        subj_start_logits, subj_end_logits = self.model(inputs)
        subj_start_logits = subj_start_logits.view(-1, self.opt['num_subj_type'] + 1)
        subj_start_type = subj_start_type.view(-1).squeeze()
        subj_start_loss = self.obj_criterion(subj_start_logits, subj_start_type).view_as(mask)
        subj_start_loss = torch.sum(subj_start_loss.mul(mask.float())) / torch.sum(mask.float())

        subj_end_loss = self.obj_criterion(subj_end_logits.view(-1, self.opt['num_subj_type'] + 1),
                                           subj_end_type.view(-1).squeeze()).view_as(mask)
        subj_end_loss = torch.sum(subj_end_loss.mul(mask.float())) / torch.sum(mask.float())

        loss = subj_start_loss + subj_end_loss

        # backward
        loss.backward()
        # torch.nn.utils.clip_grad_norm(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val

    def predict_subj_per_instance(self, words, mask):

        self.model.eval()
        hidden = self.model.based_encoder(words)

        subj_start_logits = self.model.subj_sublayer.predict_subj_start(hidden, mask)
        subj_end_logits = self.model.subj_sublayer.predict_subj_end(hidden, mask)

        return subj_start_logits, subj_end_logits

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


class SubjTypeModel(nn.Module):

    def __init__(self, opt):
        super(SubjTypeModel, self).__init__()

        self.dropout = nn.Dropout(opt['dropout'])
        self.hidden_dim = opt['word_emb_dim']

        self.linear_subj_start = nn.Linear(self.hidden_dim, opt['num_subj_type'] + 1)
        self.linear_subj_end = nn.Linear(self.hidden_dim, opt['num_subj_type'] + 1)
        self.init_weights()

    def init_weights(self):
        self.linear_subj_start.bias.data.fill_(0)
        init.xavier_uniform_(self.linear_subj_start.weight, gain=1)  # initialize linear layer

        self.linear_subj_end.bias.data.fill_(0)
        init.xavier_uniform_(self.linear_subj_end.weight, gain=1)  # initialize linear layer

    def forward(self, hidden):
        subj_start_inputs = self.dropout(hidden)
        subj_start_logits = self.linear_subj_start(subj_start_inputs)

        subj_end_inputs = self.dropout(hidden)
        subj_end_logits = self.linear_subj_end(subj_end_inputs)

        return subj_start_logits.squeeze(-1), subj_end_logits.squeeze(-1)

    def predict_subj_start(self, hidden, mask):
        subj_start_logits = self.linear_subj_start(hidden)
        subj_start_logits = torch.argmax(subj_start_logits, 2)
        subj_start_logits = subj_start_logits.mul(mask.float())

        return subj_start_logits.squeeze(-1).data.cpu().numpy().tolist()

    def predict_subj_end(self, hidden, mask):
        subj_end_logits = self.linear_subj_end(hidden)
        subj_end_logits = torch.argmax(subj_end_logits, 2)
        subj_end_logits = subj_end_logits.mul(mask.float())

        return subj_end_logits.squeeze(-1).data.cpu().numpy().tolist()


class BiLSTMCNN(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, bert_model):
        super(BiLSTMCNN, self).__init__()
        self.drop = nn.Dropout(opt['dropout'])
        self.input_size = opt['word_emb_dim']
        self.subj_sublayer = SubjTypeModel(opt)
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