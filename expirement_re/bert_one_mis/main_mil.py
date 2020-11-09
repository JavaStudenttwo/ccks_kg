# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from expirement_re.bert_one_mis.utils import *
from expirement_re.bert_one_mis.models.bert_one_mis import *
from expirement_re.bert_one_mis.config import opt
from expirement_re.bert_one_mis.dataloader import PCNNData
from torch.autograd import Variable

def collate_fn(batch):
    return batch


def test(**kwargs):
    pass


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(**kwargs):

    setup_seed(opt.seed)

    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    # torch.manual_seed(opt.seed)
    model = bert_one_mis(opt)
    if opt.use_gpu:
        # torch.cuda.manual_seed_all(opt.seed)
        model.cuda()
        # parallel
        #  model = nn.DataParallel(model)

    train_data = PCNNData(opt, train=True)
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=collate_fn)

    test_data = PCNNData(opt, train=False)
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn)
    print('train data: {}; test data: {}'.format(len(train_data), len(test_data)))

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), rho=1.0, eps=1e-6, weight_decay=opt.weight_decay)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
    # optimizer = optim.Adadelta(model.parameters(), rho=1.0, eps=1e-6, weight_decay=opt.weight_decay)
    # train
    print("start training...")
    max_pre = -1.0
    max_rec = -1.0
    for epoch in range(opt.num_epochs):

        total_loss = 0
        for idx, data in enumerate(train_data_loader):

            data = select_instance(model, data)
            label = data[3]
            model.batch_size = opt.batch_size

            optimizer.zero_grad()

            out = model(data, train=True)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch < -1:
            continue
        pro_true, total = predict(model, test_data_loader)
        # all_pre, all_rec, fp_res = eval_metric(true_y, pred_y, pred_p)

        # last_pre, last_rec = all_pre[-1], all_rec[-1]
        # if last_pre > 0.24 and last_rec > 0.24:
        # save_pr(opt.result_dir, model.model_name, epoch, all_pre, all_rec, fp_res, opt=opt.print_opt)
        # print('{} Epoch {} save pr'.format(now(), epoch + 1))
        print("save model")
        model.save(opt.print_opt)

        print('{} Epoch {}/{}: train loss: {}; test precision pro_true: {}, test recall total {}'.format(now(), epoch + 1, opt.num_epochs, total_loss, pro_true, total))


def select_instance(model, batch_data):

    model.eval()
    ent1 = []
    ent2 = []
    select_num = []
    select_lab = []
    select_sen = []
    data = []
    for idx, bag in enumerate(batch_data):
        insNum = bag[1]
        max_ins_id = 0
        # if insNum > 1:
        #     model.batch_size = insNum
        #     if opt.use_gpu:
        #         data = map(lambda x: torch.LongTensor(x).cuda(), bag)
        #     else:
        #         data = map(lambda x: torch.LongTensor(x), bag)
        #
        #     out = model(data)
        #
        #     #  max_ins_id = torch.max(torch.max(out, 1)[0], 0)[1]
        #     max_ins_id = torch.max(out[:, label], 0)[1]
        #
        #     if opt.use_gpu:
        #         #  max_ins_id = max_ins_id.data.cpu().numpy()[0]
        #         max_ins_id = max_ins_id.item()
        #     else:
        #         max_ins_id = max_ins_id.data.numpy()[0]

        max_sen = bag[4][max_ins_id]
        ent1.append(bag[0])
        ent2.append(bag[1])
        select_num.append(bag[2])
        select_lab.append(bag[3])
        select_sen.append(max_sen)

    ent1_list = np.array(utils.seq_padding(ent1))
    ent2_list = np.array(utils.seq_padding(ent2))
    select_lab = np.array(select_lab)
    sent_list = np.array(utils.seq_padding(select_sen))

    ent1_tensor = Variable(torch.LongTensor(ent1_list).cuda())
    ent2_tensor = Variable(torch.LongTensor(ent2_list).cuda())
    select_lab = Variable(torch.LongTensor(select_lab).cuda())
    sent_tensor = Variable(torch.LongTensor(sent_list).cuda())

    data.extend([ent1_tensor, ent2_tensor, select_num, select_lab, sent_tensor])

    model.train()
    return data


def predict(model, test_data_loader):

    model.eval()

    tp = 0
    tn = 0
    for idx, data in enumerate(test_data_loader):
        for bag in data:
            bag_data = []
            insNum = bag[2]

            ent1_list = np.array([bag[0]])
            ent2_list = np.array([bag[1]])
            select_lab = np.array([bag[3]])
            sent_list = np.array([bag[4][0]])

            ent1_tensor = Variable(torch.LongTensor(ent1_list).cuda())
            ent2_tensor = Variable(torch.LongTensor(ent2_list).cuda())
            true_label = Variable(torch.LongTensor(select_lab).cuda())
            sent_tensor = Variable(torch.LongTensor(sent_list).cuda())

            bag_data.extend([ent1_tensor, ent2_tensor, insNum, true_label, sent_tensor])
            out = model(bag_data)
            pred_label = torch.argmax(out, dim=1)

            if pred_label == true_label:
                tp += 1
            else:
                tn += 1

    total = len(test_data_loader.dataset)
    pro_true = tp
    model.train()
    return pro_true, total


if __name__ == "__main__":
    train()
