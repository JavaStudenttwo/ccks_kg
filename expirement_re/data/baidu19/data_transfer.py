import json
import re
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import os

# {
# "tokens":
#     ["Massachusetts", "ASTON", "MAGNA", "Great", "Barrington", ";", "also", "at",
#     "Bard", "College", ",", "Annandale-on-Hudson", ",", "N.Y.", ",", "July", "1-Aug", "."],
# "spo_list":
#     [["Annandale-on-Hudson", "/location/location/contains", "Bard College"]],
# "spo_details":
#     [[11, 12, "LOCATION", "/location/location/contains", 8, 10, "ORGANIZATION"]],
# "pos_tags":
#     ["NNP", "NNP", "NNP", "NNP", "NNP", ":", "RB", "IN", "NNP", "NNP", ",", "NNP", ",", "NNP", ",", "NNP", "NNP", "."]
# }

# {"postag": [{"word": "《", "pos": "w"}, {"word": "忘记我还是忘记他", "pos": "nw"}, {"word": "》", "pos": "w"}, {"word": "是", "pos": "v"}, {"word": "迪克牛仔", "pos": "nr"}, {"word": "于", "pos": "p"}, {"word": "2002年", "pos": "t"}, {"word": "发行", "pos": "v"}, {"word": "的", "pos": "u"}, {"word": "专辑", "pos": "n"}],
#  "text": "《忘记我还是忘记他》是迪克牛仔于2002年发行的专辑",
#  "spo_list": [{"predicate": "歌手", "object_type": "人物", "subject_type": "歌曲", "object": "迪克牛仔", "subject": "忘记我还是忘记他"}]}

# {"text": "《邪少兵王》是冰火未央写的网络小说连载于旗峰天下",
#  "spo_list": [{"predicate": "作者",
#  "object_type": {"@value": "人物"},
#  "subject_type": "图书作品",
#  "object": {"@value": "冰火未央"},
#  "subject": "邪少兵王"}]}

# {
#     S_TYPE: 娱乐人物,
#     P: 饰演,
#     O_TYPE: {
#         @value: 角色
#         inWork: 影视作品
#     }
# }
fr = open('train.txt', 'r', encoding='utf-8')

sentdic_dict = {}

for line in fr:
    ins = json.loads(line)
    sent = ins['text']
    for spo in ins['spo_list']:
        ent1 = spo['subject']
        ent2 = spo['object']
        label = spo['predicate']
        entity_key = ent1 + '-*-' + ent2 + '-*-' + label
        ent1_t = spo['subject_type']
        ent2_t = spo['object_type']
        enttype = ent1_t + '-*-' + ent2_t

    # 一个bag是一个词典
    # sentdic ['sents': , 'label': , 'entitytype': ]
    sentdic = {}
    if entity_key in sentdic_dict.keys():
        sentdic = sentdic_dict[entity_key]
        sentdic['sents'] = sentdic['sents'] + sent + '-*-'
        sentdic['entitytype'] = sentdic['entitytype'] + enttype + '-*-'
        sentdic_dict[entity_key] = sentdic
    else:
        sentdic['sents'] = sent + '-*-'
        sentdic['entitytype'] = enttype + '-*-'
        sentdic_dict[entity_key] = sentdic


with open('train_mulit.txt', 'w', encoding='utf-8') as file_obj:
    # json.dump(sentdic_dict, file_obj, ensure_ascii=False)
    for key, value in sentdic_dict.items():
        file_obj.write(key + '\n')
        sents = []
        sents_ = value['sents']
        sents = sents_.split('-*-')
        file_obj.write(str(len(sents) - 1) + '\n')
        for i in sents:
            if i != '':
                file_obj.write(i + '\n')

print('保存成功')

