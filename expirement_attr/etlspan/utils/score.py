import numpy as np
from tqdm import tqdm
from utils import loader
import json

def extract_items(bert_tokenizer, tokens_in, id2predicate, model):
    R = []
    # word2vec编码
    # _t = [word2id.get(w, 1) for w in tokens_in]
    _t = bert_tokenizer.convert_tokens_to_ids(tokens_in)

    # _c = [[char2id.get(c, 1) for c in token] for token in tokens_in]
    # _pos = loader.get_pos_tags(tokens_in, pos_tags, pos2id)

    _t = np.array(loader.seq_padding([_t]))
    # _c =  np.array(loader.char_padding([_c ]))
    # _pos = np.array(loader.seq_padding([_pos]))
    _s1, _s2, hidden = model.predict_subj_per_instance(_t)
    _s1, _s2 = np.argmax(_s1, 1), np.argmax(_s2, 1)
    for i,_ss1 in enumerate(_s1):
        if _ss1 > 0:
            _subject = ''
            for j,_ss2 in enumerate(_s2[i:]):
                if _ss2 == _ss1:
                    _subject = ''.join(tokens_in[i: i+j+1])
                    break
            if _subject:
                _k1, _k2 = np.array([[i]]), np.array([[i+j]])
                if len(tokens_in) > 150:
                    continue
                distance_to_subj = np.array([loader.get_positions(i, i+j, len(tokens_in))])
                _o1, _o2 = model.predict_obj_per_instance([_t,_k1, _k2, distance_to_subj], hidden)
                _o1, _o2 = np.argmax(_o1, 1), np.argmax(_o2, 1)
                for i,_oo1 in enumerate(_o1):
                    if _oo1 > 0:
                        for j,_oo2 in enumerate(_o2[i:]):
                            if _oo2 == _oo1:
                                _object = ''.join(tokens_in[i: i+j+1])
                                _predicate = id2predicate[_oo1]

                                #处理关系类型并解析数据

                                R.append((_subject, _predicate, _object))
                                break
    return set(R)


def evaluate(bert_tokenizer, data, id2predicate, model):
    official_A, official_B, official_C = 1e-10, 1e-10, 1e-10
    manual_A, manual_B, manual_C = 1e-10, 1e-10, 1e-10

    results = []
    for d in tqdm(iter(data)):
        R = extract_items(bert_tokenizer, d['tokens'], id2predicate, model)
        official_T = set([tuple(i) for i in d['spo_list']])
        results.append({'text':' '.join(d['tokens']), 'predict':list(R), 'truth':list(official_T)})
        official_A += len(R & official_T)
        official_B += len(R)
        official_C += len(official_T)
    return 2 * official_A / (official_B + official_C), official_A / official_B, official_A / official_C, results










    


