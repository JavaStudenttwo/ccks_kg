"""
Train a model on for JointER.
"""

import os
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import json
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/baidu19')
parser.add_argument('--vocab_dir', type=str, default='dataset/baiduRE/vocab')
parser.add_argument('--word_emb_dim', type=int, default=768, help='Word embedding dimension.')
parser.add_argument('--char_emb_dim', type=int, default=100, help='Char embedding dimension.')
parser.add_argument('--pos_emb_dim', type=int, default=20, help='Part-of-speech embedding dimension.')
parser.add_argument('--position_emb_dim', type=int, default=20, help='Position embedding dimension.')
parser.add_argument('--obj_input_dim', type=int, default=2324, help='RNN hidden state size.')
parser.add_argument('--char_hidden_dim', type=int, default=100, help='Char-CNN hidden state size.')
parser.add_argument('--num_layers', type=int, default=1, help='Num of RNN layers.')
parser.add_argument('--dropout', type=float, default=0.4, help='Input and RNN dropout rate.')
parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N embeddings.')
parser.add_argument('--subj_loss_weight', type=float, default=1, help='Subjct loss weight.')
parser.add_argument('--type_loss_weight', type=float, default=1, help='Object loss weight.')
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--lr_decay', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0, help='Applies to SGD and Adagrad.')
parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--load_saved', type=str, default='')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=50, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=10, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models/baiduRE/5_17_bert_2', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
parser.add_argument('--decode_function', type=str, default='softmax', help='使用 mycrf, softmax')

parser.add_argument('--bert_model', type=str, default='../../bert-base-chinese', help='bert模型位置')


parser.add_argument('--seed', type=int, default=35)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
args = parser.parse_args()

# 初始化bert模型
UNCASED = args.bert_model  # your path for model and vocab
VOCAB = 'vocab.txt'
bert_tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED,VOCAB))
bert_model = BertModel.from_pretrained(UNCASED)
bert_model.eval()
bert_model.cuda()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)

from expirement_er.bertcrf.utils.loader import DataLoader
from expirement_er.bertcrf.models.model import RelationModel
from expirement_er.bertcrf.utils import helper, score
from expirement_er.bertcrf.utils.vocab import Vocab


opt = vars(args)


# load data
train_data = json.load(open(opt['data_dir'] + '/train_.json', encoding='utf-8'))
dev_data = json.load(open(opt['data_dir'] + '/dev_.json', encoding='utf-8'))

subj_type2id = {
    "图书作品": 1,
    "学科专业": 2,
    "景点": 3,
    "历史人物": 4,
    "生物": 5,
    "网络小说": 6,
    "电视综艺": 7,
    "歌曲": 8,
    "机构": 9,
    "行政区": 10,
    "企业": 11,
    "影视作品": 12,
    "国家": 13,
    "书籍": 14,
    "人物": 15,
    "地点": 16,
    "学校": 17,
    "网站": 18,
    "目": 19,
    "作品": 20,
    "音乐专辑": 21,
    "城市": 22,
    "Text": 23,
    "气候": 24,
    "Date": 25,
    "语言": 26,
    "Number": 27,
    "出版社": 28
}
english_entity_type_vs_chinese_entity_type = {v: k for k, v in subj_type2id.items()}

START_TAG = "[CLS]"
END_TAG = "[SEP]"
O = "O"
B1 = "B-1"
I1 = "I-1"
B2 = "B-2"
I2 = "I-2"
B3 = "B-3"
I3 = "I-3"
B4 = "B-4"
I4 = "I-4"
B5 = "B-5"
I5 = "I-5"
B6 = "B-6"
I6 = "I-6"
B7 = "B-7"
I7 = "I-7"
B8 = "B-8"
I8 = "I-8"
B9 = "B-9"
I9 = "I-9"
B10 = "B-10"
I10 = "I-10"
B11 = "B-11"
I11 = "I-11"
B12 = "B-12"
I12 = "I-12"
B13 = "B-13"
I13 = "I-13"
B14 = "B-14"
I14 = "I-14"
B15 = "B-15"
I15 = "I-15"
B16 = "B-16"
I16 = "I-16"
B17 = "B-17"
I17 = "I-17"
B18 = "B-18"
I18 = "I-18"
BPeople = "B-People"
IPeople = "I-People"
BIndustry = "B-Industry"
IIndustry = "I-Industry"
BBusiness = 'B-Business'
IBusiness = 'I-Business'
BProduct = 'B-Product'
IProduct = 'I-Product'
BReport = 'B-Report'
IReport = 'I-Report'
BOrganization = 'B-Organization'
IOrganization = 'I-Organization'
BRisk = 'B-Risk'
IRisk = 'I-Risk'
BArticle = 'B-Article'
IArticle = 'I-Article'
BIndicator = 'B-Indicator'
IIndicator = 'I-Indicator'
BBrand = 'B-Brand'
IBrand = 'I-Brand'

PAD = "[PAD]"
UNK = "[UNK]"
tag2idx = {
    PAD: 0,
    END_TAG: 1,
    O: 2,
    BPeople: 3,
    IPeople: 4,
    BIndustry: 5,
    IIndustry: 6,
    BBusiness: 7,
    IBusiness: 8,
    BProduct: 9,
    IProduct: 10,
    BReport: 11,
    IReport: 12,
    BOrganization: 13,
    IOrganization: 14,
    BRisk: 15,
    IRisk: 16,
    BArticle: 17,
    IArticle: 18,
    BIndicator: 19,
    IIndicator: 20,
    BBrand: 21,
    IBrand: 22,
    B1: 23,
    I1: 24,
    B2 :  25,
    I2 :  26,
    B3 :  27,
    I3 :  28,
    B4 :  29,
    I4 :  30,
    B5 :  31,
    I5 :  32,
    B6 :  33,
    I6 :  34,
    B7 :  35,
    I7 :  36,
    B8 :  37,
    I8 :  38,
    B9 :  39,
    I9 :  40,
    B10 :  41,
    I10 :  42,
    B11 :  43,
    I11 :  44,
    B12 :  45,
    I12 :  46,
    B13 :  47,
    I13 :  48,
    B14 :  49,
    I14 :  50,
    B15 :  51,
    I15 :  52,
    B16 :  53,
    I16 :  54,
    B17 :  55,
    I17 :  56,
    B18 :  57,
    I18 :  58,
    START_TAG: 59,
    UNK: 60,
}
tag2id = tag2idx
idx2tag = {v: k for k, v in tag2idx.items()}
opt['num_tags'] = len(tag2idx)


# load data 加载数据
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(train_data, subj_type2id, opt['batch_size'])

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)


# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\dev_p\tdev_r\tdev_f1")
# print model info
helper.print_config(opt)
# model 初始化模型
model = RelationModel(opt, bert_model)
if opt['load_saved'] != '':
    model.load(opt['save_dir']+'/'+opt['load_saved']+'/best_model.pt')
dev_f1_history = []
current_lr = opt['lr']


global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'

# start training 训练
for epoch in range(1, opt['num_epoch']+1):
    train_loss = 0
    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1
        loss = model.update(batch)
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            max_steps = len(train_batch) * opt['num_epoch']
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))

    # eval on dev
    print("Evaluating on dev set...")
    dev_f1, dev_p, dev_r, results = score.evaluate(opt, dev_data, model, idx2tag)

    train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
    best_f1 = dev_f1 if epoch == 1 or dev_f1 > max(dev_f1_history) else max(dev_f1_history)
    print("epoch {}: train_loss = {:.6f}, dev_p = {:.6f}, dev_r = {:.6f}, dev_f1 = {:.4f}, best_f1 = {:.4f}".format(epoch,\
            train_loss, dev_p, dev_r, dev_f1, best_f1))
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_p, dev_r, dev_f1))

    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    model.save(model_file, epoch)
    if epoch == 1 or dev_f1 > max(dev_f1_history):
        copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")
        with open(model_save_dir + '/best_dev_results.json', 'w', encoding='utf-8') as fw:
            json.dump(results, fw, indent=4, ensure_ascii=False)
        print("new best results saved.")
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)
    
    # lr schedule
    if len(dev_f1_history) > 10 and dev_f1 <= dev_f1_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad']:
        current_lr *= opt['lr_decay']
        model.update_lr(current_lr)

    dev_f1_history += [dev_f1]
    print("")

print("Training ended with {} epochs.".format(epoch))

