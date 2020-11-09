# # 用hanlp进行实体识别
#
# hanlp支持对人物、机构的实体识别，可以使用它来对其中的两个实体类型进行识别：人物、机构。
#
# hanlp见[https://github.com/hankcs/HanLP](https://github.com/hankcs/HanLP)

# In[ ]:

import hanlp
import re


## NER by third party tool
class HanlpNER:
    def __init__(self):
        self.recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
        self.max_sent_len = 126
        self.ent_type_map = {
            'NR': '人物',
            'NT': '机构'
        }
        self.black_list = {'公司'}

    def recognize(self, sent):
        entities_dict = {}
        for result in self.recognizer.predict([list(sent)]):
            for entity, hanlp_ent_type, _, _ in result:
                if not re.findall(r'^[\.\s\da-zA-Z]{1,2}$', entity) and \
                        len(entity) > 1 and entity not in self.black_list \
                        and hanlp_ent_type in self.ent_type_map:
                    entities_dict.setdefault(self.ent_type_map[hanlp_ent_type], []).append(entity)
        return entities_dict
