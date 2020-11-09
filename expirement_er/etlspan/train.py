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
parser.add_argument('--data_dir', type=str, default='dataset/baidu')
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
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=50, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=10, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models/baiduRE/5_17_bert_2', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

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

from expirement_er.etlspan.utils.loader import DataLoader
from expirement_er.etlspan.models.model import RelationModel
from expirement_er.etlspan.utils import helper, score
from expirement_er.etlspan.utils.vocab import Vocab


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
opt['num_subj_type'] = len(subj_type2id)


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
    dev_f1, dev_p, dev_r, results = score.evaluate(dev_data, model)

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

