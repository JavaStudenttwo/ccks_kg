import tqdm
from pytorch_pretrained_bert import BertModel, BertTokenizer
from utils.schemas import *
import re
from utils.QuestionText import *


# ## 预处理数据函数
#
# `preprocess_data` 函数中的 `for_train` 参数比较重要，指示是否是训练集
#
# 由于给定的训练数据实体部分没有给定出现的位置，这里需要自行查找到实体出现的位置
#
# - 如果是训练集, 按照`entities_json`中的内容在文章中寻找位置并标注, 并将训练数据处理成bio形式
# - 测试数据仅仅做了分句并转化成token id

# In[ ]:


class Article:
    def __init__(self, text):
        self._text = text
        self.para_texts = self.split_into_paras(self._text)
        self.sent_texts = [self.split_into_sentence(para) for para in self.para_texts]

    def fix_text(self, text: str) -> str:
        paras = text.split('\n')
        paras = list(filter(lambda para: len(para.strip()) != 0, paras))
        return '\n'.join(paras)

    def split_into_paras(self, text: str):
        paras = list(filter(lambda para: len(para.strip()) != 0, text.split('\n')))
        return paras

    def split_into_sentence(self, one_para_text: str, splited_puncs=None):
        if splited_puncs is None:
            splited_puncs = ['。', '？', '！']
        splited_re_pattern = '[' + ''.join(splited_puncs) + ']'

        para = one_para_text
        sentences = re.split(splited_re_pattern, para)
        sentences = list(filter(lambda sent: len(sent) != 0, sentences))

        return sentences

    def find_sents_by_entity_name(self, entity_text):
        ret_sents = []
        if entity_text not in self._text:
            return []
        else:
            for para in self.split_into_paras(self._text):
                if entity_text not in para:
                    continue
                else:
                    for sent in self.split_into_sentence(para):
                        if entity_text in sent:
                            ret_sents.append(sent)
        return ret_sents


def preprocess_data(entities_json_,
                    article_texts,
                    tokenizer: BertTokenizer,
                    for_train: bool = True):
    """
    [{
        'sent': xxx, 'entity_name': yyy, 'entity_type': zzz, 'start_token_id': 0, 'end_token_id': 5,
        'start_index': 0, 'end_index': 2,
            'sent_tokens': ['token1', 'token2'], 'entity_tokens': ['token3', 'token4']
    }]
    """
    entities_json = {}
    entities_json["指标"] = entities_json_["指标"]
    entities_json["品牌"] = entities_json_["品牌"]
    entities_json["行业"] = entities_json_["行业"]
    entities_json["业务"] = entities_json_["业务"]
    entities_json["产品"] = entities_json_["产品"]

    preprocessed_datas = []

    all_sents = []

    # 获得bert词表中的所有字符
    dict_list = []
    dict_ = tokenizer.vocab
    for i in dict_:
        dict_list.append(i)

    # 对问题进行处理
    query_dict = {}
    for entity_type, query in type2query.items():
        query_tokens = ['[CLS]']
        query_tokens.extend(list(query))
        query_tokens.append('[SEP]')
        query_dict[entity_type] = query_tokens

    # 为每一篇研报创建一个Article类，进行文章处理
    for article in tqdm.tqdm([Article(t) for t in article_texts]):
        # 将文件分为一个个段落
        for para_text in article.para_texts:
            # 将段落分为一个个句子
            for sent in article.split_into_sentence(para_text):
                sents_tokens = list(sent)
                sents_tokens.append('[SEP]')

                # 构建训练集和验证集
                if for_train:
                    # 根据实体类型遍历句子，使用远程监督方法标注语料
                    for entity_type, entities in entities_json.items():
                        sent_tokens = []
                        sent_tokens.extend(query_dict[entity_type])
                        sent_tokens.extend(sents_tokens)
                        query_len = len(query_dict[entity_type])

                        entity_labels = []
                        for entity_name in entities:
                            if entity_name not in sent:
                                continue
                            all_sents.append(sent)
                            start_end_indexes = _find_all_start_end(sent, entity_name)
                            assert len(start_end_indexes) >= 1
                            for str_start_index, str_end_index in start_end_indexes:
                                entity_tokens = list(entity_name)

                                one_entity_label = {
                                    'entity_type': entity_type,
                                    'start_token_id': str_start_index,
                                    'end_token_id': str_end_index,
                                    'start_index': str_start_index + query_len,
                                    'end_index': str_end_index + query_len,
                                    'entity_tokens': entity_tokens,
                                    'entity_name': entity_name
                                }
                                entity_labels.append(one_entity_label)

                        # 遍历entity_labels，处理一个句子中的所有实体
                        ts1, ts2 = [0] * len(sent_tokens), [0] * len(sent_tokens)
                        if entity_labels is not None:
                            for j in entity_labels:
                                stp = entity_type2id[j['entity_type']]
                                ts1[j['start_index']] = stp
                                ts2[j['end_index']] = stp

                        # 直接根据字符串分字得到的字序列不能直接使用bert转码，需要进行一个预处理将bert词表中没有的字符设为'[UNK]'
                        sent_tokens_ = []
                        for token in sent_tokens:
                            if token not in dict_list:
                                sent_tokens_.append('[UNK]')
                            else:
                                sent_tokens_.append(token)

                        # 使用bert词表转码
                        sent_token_ids = tokenizer.convert_tokens_to_ids(sent_tokens_)
                        # 处理完成，将需要的信息进行整理
                        preprocessed_datas.append({
                            'sent': sent,
                            'sent_tokens': sent_tokens,
                            'sent_token_ids': sent_token_ids,
                            'entity_labels': entity_labels,
                            'ts1': ts1,
                            'ts2': ts2
                        })
                # 构建测试集
                else:
                    for entity_type in entities_json.keys():
                        sent_tokens = []
                        sent_tokens.extend(query_dict[entity_type])
                        sent_tokens.extend(sents_tokens)

                        ts1, ts2 = [0] * len(sent_tokens), [0] * len(sent_tokens)
                        # 直接根据字符串分字得到的字序列不能直接使用bert转码，需要进行一个预处理将bert词表中没有的字符设为'[UNK]'
                        sent_tokens_ = []
                        for token in sent_tokens:
                            if token not in dict_list:
                                sent_tokens_.append('[UNK]')
                            else:
                                sent_tokens_.append(token)

                        # 使用bert词表转码
                        sent_token_ids = tokenizer.convert_tokens_to_ids(sent_tokens_)
                        # 处理完成，将需要的信息进行整理
                        preprocessed_datas.append({
                            'sent': sent,
                            'sent_tokens': sent_tokens,
                            'sent_token_ids': sent_token_ids,
                            'entity_labels': [],
                            'ts1': ts1,
                            'ts2': ts2
                        })

    return preprocessed_datas


def _find_all_start_end(source, target):
    if not target:
        return []
    occurs = []
    offset = 0
    while offset < len(source):
        found = source[offset:].find(target)
        if found == -1:
            break
        else:
            occurs.append([offset + found, offset + found + len(target) - 1])
        offset += (found + len(target))
    return occurs
