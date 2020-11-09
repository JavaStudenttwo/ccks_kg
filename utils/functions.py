import tqdm
from collections import defaultdict
import torch
import json
from pytorch_pretrained_bert import BertModel, BertTokenizer
import jieba
from jieba.analyse.tfidf import TFIDF
from jieba.posseg import POSTokenizer

from model_entity.HanlpNER import HanlpNER
from utils.preprocessing import *
from pathlib import Path
from utils.MyDataset import MyDataset
from parameters import *

# In[ ]:


# # 预训练模型配置
#
# 参考 https://github.com/huggingface/pytorch-transformers 下载预训练模型，并配置下面参数为相关路径
#
# ```python
# PRETRAINED_BERT_MODEL_DIR = '/you/path/to/bert-base-chinese/'
# ```

# In[ ]:


# # 一些参数

# In[ ]:


DATA_DIR = opt['data_dir']  # 输入数据文件夹
OUT_DIR = opt['out_dir']  # 输出文件夹

Path(OUT_DIR).mkdir(exist_ok=True)

BATCH_SIZE = opt['batch_size']
TOTAL_EPOCH_NUMS = opt['total_epoch_nums']

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'
YANBAO_DIR_PATH = str(Path(DATA_DIR, 'yanbao_txt'))
SAVE_MODEL_DIR = str(OUT_DIR)


def aug_entities_by_third_party_tool():
    hanlpner = HanlpNER()
    entities_by_third_party_tool = defaultdict(list)
    for file in tqdm.tqdm(list(Path(DATA_DIR, 'yanbao_txt').glob('*.txt'))[:]):
        with open(file, encoding='utf-8') as f:
            sents = [[]]
            cur_sent_len = 0
            for line in f:
                for sent in split_to_subsents(line):
                    sent = sent[:hanlpner.max_sent_len]
                    if cur_sent_len + len(sent) > hanlpner.max_sent_len:
                        sents.append([sent])
                        cur_sent_len = len(sent)
                    else:
                        sents[-1].append(sent)
                        cur_sent_len += len(sent)
            sents = [''.join(_) for _ in sents]
            sents = [_ for _ in sents if _]
            for sent in sents:
                entities_dict = hanlpner.recognize(sent)
                for ent_type, ents in entities_dict.items():
                    entities_by_third_party_tool[ent_type] += ents

    for ent_type, ents in entities_by_third_party_tool.items():
        entities_by_third_party_tool[ent_type] = list([ent for ent in set(ents) if len(ent) > 1])
    return entities_by_third_party_tool


def custom_collate_fn(data):
    # copy from torch official，无需深究
    from torch._six import container_abcs, string_classes

    r"""Converts each NumPy array data field into a tensor"""
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, container_abcs.Mapping):
        tmp_dict = {}
        for key in data:
            if key in ['sent_token_ids', 'tag_ids', 'mask']:
                tmp_dict[key] = custom_collate_fn(data[key])
                if key == 'mask':
                    tmp_dict[key] = tmp_dict[key].byte()
            else:
                tmp_dict[key] = data[key]
        return tmp_dict
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(custom_collate_fn(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
        return [custom_collate_fn(d) for d in data]
    else:
        return data


def read_json(file_path):
    with open(file_path, mode='r', encoding='utf8') as f:
        return json.load(f)


def build_dataloader(preprocessed_datas, tokenizer: BertTokenizer, batch_size=32, shuffle=True):
    dataset = MyDataset(preprocessed_datas, tokenizer)
    import torch.utils.data
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=shuffle)
    return dataloader


# ### 模型预测结果后处理函数
#
# - `review_model_predict_entities`函数将模型预测结果后处理，从而生成提交文件格式


def review_model_predict_entities(model_predict_entities):
    word_tag_map = POSTokenizer().word_tag_tab
    idf_freq = TFIDF().idf_freq
    reviewed_entities = defaultdict(list)
    for ent_type, ent_and_sent_list in model_predict_entities.items():
        for ent, sent in ent_and_sent_list:
            start = sent.lower().find(ent)
            if start == -1:
                continue
            start += 1
            end = start + len(ent) - 1
            tokens = jieba.lcut(sent)
            offset = 0
            selected_tokens = []
            for token in tokens:
                offset += len(token)
                if offset >= start:
                    selected_tokens.append(token)
                if offset >= end:
                    break

            fixed_entity = ''.join(selected_tokens)
            fixed_entity = re.sub(r'\d*\.?\d+%$', '', fixed_entity)
            if ent_type == '人物':
                if len(fixed_entity) >= 10:
                    continue
            if len(fixed_entity) <= 1:
                continue
            if re.findall(r'^\d+$', fixed_entity):
                continue
            if word_tag_map.get(fixed_entity, '') == 'v' and idf_freq[fixed_entity] < 7:
                continue
            reviewed_entities[ent_type].append(fixed_entity)
    return reviewed_entities