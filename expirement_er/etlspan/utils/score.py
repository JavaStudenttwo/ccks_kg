import numpy as np
from tqdm import tqdm
from expirement_er.etlspan.utils import loader
import json
from torch import nn
import torch


def extract_items(tokens, tokens_ids, model):
    R = []
    if len(tokens_ids) > 180:
        return set(R)

    _t = np.array(loader.seq_padding([tokens_ids]))

    _s1, _s2, hidden = model.predict_subj_per_instance(_t)

    _s1 = _s1.tolist()
    _s2 = _s2.tolist()

    for i, _ss1 in enumerate(_s1):
        if _ss1 > 0:
            for j, _ss2 in enumerate(_s2[i:]):
                if _ss2 == _ss1:
                    entity = ''.join(tokens[i: i + j + 1])
                    R.append(entity)
                    break
    return set(R)


def evaluate(data, model):
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
    for d in tqdm(iter(data)):
        R = extract_items(d['sent_tokens'], d['sent_token_ids'], model)
        # official_T = set([tuple(i['entity_name']) for i in d['spo_details']])
        official_T_list = []
        for i in d['entity_labels']:
            official_T_list.append(i['entity_name'])
        official_T = set(official_T_list)
        results.append({'text': ''.join(d['sent_tokens']), 'predict': list(R), 'truth': list(official_T)})
        official_A += len(R & official_T)
        official_B += len(R)
        official_C += len(official_T)
    return 2 * official_A / (official_B + official_C), official_A / official_B, official_A / official_C, results













