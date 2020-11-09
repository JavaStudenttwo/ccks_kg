import numpy as np
from tqdm import tqdm
from expirement_er.etlspan.utils import loader
import json

def extract_items(bert_tokenizer, tokens_in, id2predicate, model):

    result = []
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
            R = []
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
                                R.append({'subject':_subject, 'predicate':_predicate, 'object':_object})
                                break
            for sop in R:
                # 以主体为单位，处理关系类型并解析数据
                predicate_list = []
                predicate_list = sop['predicate'].split('-')
                subject_value = sop['subject']
                object_value = sop['object']

                # "父亲-人物-@value-人物-normal"
                # "spo_list": [
                #     {
                #         "predicate": "歌手",
                #         "object_type": {
                #             "@value": "人物"
                #         },
                #         "subject_type": "歌曲",
                #         "object": {
                #             "@value": "龚琳娜"
                #         },
                #         "subject": "忐忑"
                #     }
                # ]
                # 如果是正常三元组
                if predicate_list.index(4) == 'normal':
                    predicate_value = predicate_list.index(0)
                    subject_type = predicate_list.index(1)
                    object_type = predicate_list.index(3)
                    # 组合简单o值的三元组结果
                    result.append(
                        {
                            "predicate": predicate_value,
                            "object_type": {
                                "@value": object_type
                            },
                            "subject_type": subject_type,
                            "object": {
                                "@value": object_value
                            },
                            "subject": subject_value
                        }
                    )

                # 如果是重叠三元组
                elif predicate_list.index(4) == 'complex' and predicate_list.index(2) == '@value':

                    predicate_value = predicate_list.index(0)
                    subject_type = predicate_list.index(1)
                    object_type = predicate_list.index(3)

                    object_value_, object_type_ = {}
                    object_value_['@value'] = object_value
                    object_type_['@value'] = object_type

                    # "父亲-人物-@value-人物-normal"
                    for sop_ in R:
                        predicate_list_ = sop_['predicate'].split('-')
                        if predicate_list_.index(4) == 'complex' and predicate_list_.index(2) != '@value':
                            key_ = predicate_list_.index(2)

                            object_value_[key_] = sop_['object']
                            object_type_[key_] = predicate_list_.index(3)
                    # 组合复杂o值的三元组结果
                    result.append(
                        {
                            "predicate": predicate_value,
                            "object_type": object_type_,
                            "subject_type": subject_type,
                            "object": object_value_,
                            "subject": subject_value
                        }
                    )
                else:
                    continue



    return result



def evaluate(bert_tokenizer, data, id2predicate, model):

    results = []
    for d in tqdm(iter(data)):
        result_ = extract_items(bert_tokenizer, d['tokens'], id2predicate, model)
        results.append({'text': d['text'], 'spo_list': result_, 'real_result': d['spo_list']})

    return results


# [
#     {
#         "text": "1、龚琳娜，《忐忑》，一经播放，瞬间成为神曲的代表之作，全曲没有一句正经歌词，不是啊就是哦，也正因为这首歌，才奠定了她在歌坛的位置",
#         "spo_list": [
#             {
#                 "predicate": "歌手",
#                 "object_type": {
#                     "@value": "人物"
#                 },
#                 "subject_type": "歌曲",
#                 "object": {
#                     "@value": "龚琳娜"
#                 },
#                 "subject": "忐忑"
#             }
#         ]
#     }
# ]

#      {
#         "text": "1992年春晚小品《我想有个家》（赵本山、黄晓娟）获戏剧类一等奖，经典台词：“男人的一半是女人”“我叫不紧张”",
#         "spo_list": [
#             {
#                 "predicate": "获奖",
#                 "object_type": {
#                     "inWork": "作品",
#                     "onDate": "Date",
#                     "@value": "奖项"
#                 },
#                 "subject_type": "娱乐人物",
#                 "object": {
#                     "inWork": "我想有个家",
#                     "onDate": "1992年",
#                     "@value": "获戏剧类一等奖"
#                 },
#                 "subject": "赵本山"
#             }
#         ]
#     }










    


