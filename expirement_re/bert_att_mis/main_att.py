# -*- coding: utf-8 -*-

from expirement_re.bert_att_mis.config import opt
from expirement_re.bert_att_mis.models.bert_att_mis import *
import torch
from expirement_re.bert_att_mis.dataloader import *
import torch.optim as optim
from expirement_re.bert_att_mis.utils import save_pr, now, eval_metric
from torch.utils.data import DataLoader


def collate_fn(batch):
    return batch


def test(**kwargs):
    pass


def train(**kwargs):

    kwargs.update({'model': 'bert_att_mis'})
    opt.parse(kwargs)

    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = bert_att_mis(opt)
    if opt.use_gpu:
        model.cuda()

    # loading data
    train_data = PCNNData(opt, train=True)
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=collate_fn)

    test_data = PCNNData(opt, train=False)
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn)
    print('{} train data: {}; test data: {}'.format(now(), len(train_data), len(test_data)))

    # criterion and optimizer
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6)

    # train
    #  max_pre = -1.0
    #  max_rec = -1.0
    for epoch in range(opt.num_epochs):
        total_loss = 0
        for idx, data in enumerate(train_data_loader):

            label = [l[3] for l in data]

            optimizer.zero_grad()
            model.batch_size = opt.batch_size
            loss = model(data, label)
            if opt.use_gpu:
                label = torch.LongTensor(label).cuda()
            else:
                label = torch.LongTensor(label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # true_y, pred_y, pred_p= predict(model, test_data_loader)
        # all_pre, all_rec = eval_metric(true_y, pred_y, pred_p)
        correct, positive_num = predict_var(model, test_data_loader)
        # last_pre, last_rec = eval_metric_var(pred_res, p_num)

        # if pred_res > 0.8 and p_num > 0.8:
            # save_pr(opt.result_dir, model.model_name, epoch, last_pre, last_rec, opt=opt.print_opt)
            # print('{} Epoch {} save pr'.format(now(), epoch + 1))

        print('{} Epoch {}/{}: train loss: {}; test correct: {}, test num {}'.format(now(), epoch + 1, opt.num_epochs, total_loss, correct, positive_num))


def predict_var(model, test_data_loader):

    model.eval()

    res = []
    true_y = []

    correct = 0
    for idx, data in enumerate(test_data_loader):
        labels = [l[3] for l in data]
        out = model(data)
        true_y.extend(labels)
        if opt.use_gpu:
            #  out = map(lambda o: o.data.cpu().numpy().tolist(), out)
            out = out.data.cpu().numpy().tolist()
        else:
            #  out = map(lambda o: o.data.numpy().tolist(), out)
            out = out.data.numpy().tolist()

        for i, j in zip(out, labels):
            if i == j:
                correct += 1

    model.train()
    positive_num = len(test_data_loader.dataset.x)
    return correct, positive_num


def eval_metric_var(pred_res, p_num):
    correct = 0.0
    p_nums = 0

    for i in pred_res:
        true_y = i[0]
        pred_y = i[1]
        values = i[2]
        if values > 0.5:
            p_nums += 1
            if true_y == pred_y:
                correct += 1

    if p_nums == 0:
        precision = 0
    else:
        precision = correct / p_nums
    recall = correct / p_num

    print("positive_num: {};  correct: {}".format(p_num, correct))
    return precision, recall


def predict(model, test_data_loader):

    model.eval()

    pred_y = []
    true_y = []
    pred_p = []
    for idx, data in enumerate(test_data_loader):
        labels = [l[3] for l in data]
        true_y.extend(labels)
        out = model(data)
        res = torch.max(out, 1)
        if model.opt.use_gpu:
            pred_y.extend(res[1].data.cpu().numpy().tolist())
            pred_p.extend(res[0].data.cpu().numpy().tolist())
        else:
            pred_y.extend(res[1].data.numpy().tolist())
            pred_p.extend(res[0].data.numpy().tolist())
        # if idx % 100 == 99:
            # print('{} Eval: iter {}'.format(now(), idx))

    size = len(test_data_loader.dataset)
    assert len(pred_y) == size and len(true_y) == size
    assert len(pred_y) == len(pred_p)
    model.train()
    return true_y, pred_y, pred_p


if __name__ == "__main__":
    train()
