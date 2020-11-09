"""
Run evaluation with saved models.
"""

import os
import random
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import json
from models.model import RelationModel
from utils import torch_utils, helper, result
from utils.vocab import Vocab
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM



parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/NYT-multi/data')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
model = RelationModel(opt)
model.load(model_file)



# load data
data_file = args.data_dir + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
data = json.load(open(data_file))

id2predicate, predicate2id, id2subj_type, subj_type2id, id2obj_type, obj_type2id = json.load(open(opt['data_dir'] + '/schemas.json'))
id2predicate = {int(i):j for i,j in id2predicate.items()}

# 加载bert词表
UNCASED = args.bert_model  # your path for model and vocab
VOCAB = 'bert-base-chinese-vocab.txt'
bert_tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED,VOCAB))

helper.print_config(opt)

results = result.evaluate(bert_tokenizer, data, id2predicate, model)
results_save_dir = opt['model_save_dir'] + '/results.json'
print("Dumping the best test results to {}".format(results_save_dir))

with open(results_save_dir, 'w') as fw:
    json.dump(results, fw, indent=4, ensure_ascii=False)

print("Evaluation ended.")

