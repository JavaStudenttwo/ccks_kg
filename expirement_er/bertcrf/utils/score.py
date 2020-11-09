import numpy as np
from tqdm import tqdm
from expirement_er.bertcrf.utils import loader
import json
from torch import nn
import torch


def extract_items(opt, tokens, tokens_ids, model, idx2tag):
    R = []

    _t = np.array(loader.seq_padding([tokens_ids]))

    best_tags_list_ids = model.predict_subj_per_instance(_t)
    # best_tags_list_ids = best_tags_list_ids.squeeze(-1).cpu().detach().numpy().tolist()[0]
    best_tags_list_ids = best_tags_list_ids[0]
    # best_tags_list = [[idx2tag[idx] for idx in idxs] for idxs in best_tags_list_ids]

    record = False
    for token_index, tag_id in enumerate(best_tags_list_ids):
        tag = idx2tag[tag_id]
        if tag.startswith('B'):
            start_token_index = token_index
            record = True
        elif record and tag == 'O':
            end_token_index = token_index
            str_start_index = start_token_index
            str_end_index = end_token_index
            # 使用crf时多了一个起始标签
            if opt['decode_function'] == 'mycrf':
                entity_name = tokens[str_start_index + 1: str_end_index + 1]
            elif opt['decode_function'] == 'softmax':
                entity_name = tokens[str_start_index: str_end_index]
            entity_name = ''.join(entity_name)
            R.append(entity_name)
            record = False
    # if R == []:
    #     R.append(" ")
    return set(tuple(R))


def evaluate(opt, data, model, idx2tag):
    official_A, official_B, official_C = 1e-10, 1e-10, 1e-10
    manual_A, manual_B, manual_C = 1e-10, 1e-10, 1e-10

    # {
    # "sent": "半导体行情的风险是什么",
    # "sent_tokens": ["半", "导", "体", "行", "情", "的", "风", "险", "是", "什", "么"],
    # "sent_token_ids": [1288, 2193, 860, 6121, 2658, 4638, 7599, 7372, 3221, 784, 720],
    # "entity_labels": [{"entity_type": "研报", "start_token_id": 0, "end_token_id": 10, "start_index": 0, "end_index": 10,
    # "entity_tokens": ["半", "导", "体", "行", "情", "的", "风", "险", "是", "什", "么"],
    # "entity_name": "半导体行情的风险是什么"}],
    # "tags": ["B-Report", "I-Report", "I-Report", "I-Report", "I-Report", "I-Report", "I-Report", "I-Report", "I-Report", "I-Report", "I-Report"],
    # "tag_ids": [11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]},

    results = []
    index = 0
    for d in tqdm(iter(data)):
        # index += 1
        # if index > 200:
        #     break
        if len(d['sent_token_ids']) > 180:
            continue
        R = extract_items(opt, d['sent_tokens'], d['sent_token_ids'], model, idx2tag)
        # official_T = set([tuple(i['entity_name']) for i in d['spo_details']])
        official_T_list = []
        for i in d['entity_labels']:
            official_T_list.append(i['entity_name'])
        official_T = set(tuple(official_T_list))
        results.append({'text': ''.join(d['sent_tokens']), 'predict': list(R), 'truth': list(official_T)})
        official_A += len(R & official_T)
        official_B += len(R)
        official_C += len(official_T)
    return 2 * official_A / (official_B + official_C), official_A / official_B, official_A / official_C, results













