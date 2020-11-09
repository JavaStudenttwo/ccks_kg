# -*- coding: utf-8 -*-
label2id_dict = {
    "妻子": 1,
    "成立日期": 2,
    "号": 3,
    "出生地": 4,
    "注册资本": 5,
    "作曲": 6,
    "歌手": 7,
    "出品公司": 8,
    "人口数量": 9,
    "连载网站": 10,
    "创始人": 11,
    "首都": 12,
    "民族": 13,
    "目": 14,
    "邮政编码": 15,
    "毕业院校": 16,
    "作者": 17,
    "母亲": 18,
    "所在城市": 19,
    "制片人": 20,
    "出生日期": 21,
    "作词": 22,
    "占地面积": 23,
    "主演": 24,
    "面积": 25,
    "嘉宾": 26,
    "总部地点": 27,
    "修业年限": 28,
    "编剧": 29,
    "导演": 30,
    "主角": 31,
    "上映时间": 32,
    "出版社": 33,
    "祖籍": 34,
    "董事长": 35,
    "朝代": 36,
    "海拔": 37,
    "父亲": 38,
    "身高": 39,
    "主持人": 40,
    "改编自": 41,
    "简称": 42,
    "国籍": 43,
    "所属专辑": 44,
    "丈夫": 45,
    "气候": 46,
    "官方语言": 47,
    "字": 48,
    "无关系": 49
}
id2label_dict = {v: k for k, v in label2id_dict.items()}


class DefaultConfig(object):
    label2id = label2id_dict
    id2label = id2label_dict

    model = 'bert_one_mis'  # the name of used model, in  <models/__init__.py>

    root_path = '../data'
    result_dir = './out'

    bert_tokenizer_path = '../../bert-base-chinese/vocab.txt'
    bert_path = '../../bert-base-chinese'

    load_model_path = 'checkpoints/model.pth'  # the trained model

    seed = 3435
    batch_size = 10  # batch size
    use_gpu = True  # user GPU or not
    gpu_id = 0
    num_workers = 0  # how many workers for loading data

    hidden_dim = 768
    rel_num = len(label2id_dict)

    num_epochs = 16  # the number of epochs for training
    drop_out = 0.5
    lr = 2e-5  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0.0001  # optimizer parameter

    print_opt = 'DEF'


def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        print('*************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')


DefaultConfig.parse = parse
opt = DefaultConfig()
