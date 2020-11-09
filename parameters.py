import argparse
from utils.schemas import *
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--lr_decay', type=float, default=0)
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N embeddings.')
parser.add_argument('--word_emb_dim', type=int, default=768, help='Word embedding dimension.')
parser.add_argument('--dropout', type=float, default=0.4, help='Input and RNN dropout rate.')
parser.add_argument('--weight_decay', type=float, default=0, help='Applies to SGD and Adagrad.')
parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
parser.add_argument('--bert_model', type=str, default='./bert-base-chinese', help='bert模型位置')
parser.add_argument('--data_dir', type=str, default='./data/ccks2020-stage2-open', help='输入数据文件位置')
parser.add_argument('--out_dir', type=str, default='./output/6.17', help='输出模型文件位置')
parser.add_argument('--out_data_dir', type=str, default='./output/data', help='输出数据文件位置')
parser.add_argument('--batch_size', type=int, default=6, help='batch_size')
parser.add_argument('--total_epoch_nums', type=int, default=5, help='epoch')
parser.add_argument('--nums_round', type=int, default=10, help='nums_round')
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--seed', type=int, default=35)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
args = parser.parse_args()
opt = vars(args)

opt['num_subj_type'] = len(entity_type2id)