import json
from regulation import *
from extract_attrs import *
from extract_relations import *
from extract_entities import *


fr = open('answers11.json', 'r', encoding='utf-8')

ins = json.load(fr)
dict__ = {
    "品牌": ["魅族"],
    "机构": ["日本厚生劳动省"],
}

dict_ = {
    '行业': [],
    '机构': ["官网", "三环", "学习中心", "营运中心", "维达团队", "发改", "保险", "云商", "解放军海", "沪深", "日本厚生劳动", "人保", "招商", "新加", "研究院", "美国总部", "中百", "市场", "移为", "太阳", "云计算", "田忌", "腾讯系", "工程学院", "平台", "上证报", "中国太平", "字节", "亚太", "两颗", "2颗鸡", "件公司", "我厨", "2018", "201",],
    '研报': [],
    '指标': ["– "],
    '人物': [],
    '业务': [],
    '风险': [],
    '文章': [],
    '品牌': [],
    '产品': []
}

submit_entities = {}

for key, value in ins['entities'].items():
    submit_entities[key] = list(set(ins['entities'][key]).difference(set(dict_[key])))


# for key, value in submit_entities.items():
#     for key__, value__ in dict__.items():
#         if key__ == key:
#             submit_entities[key] = list(set(submit_entities[key]).union(set(dict__[key])))
#         else:
#             submit_entities[key] = submit_entities[key]

# submit_entities = ins['entities']
# 使用规则匹配得到实体属性
train_attrs = read_json(Path(DATA_DIR, 'attrs.json'))['attrs']
submit_attrs = extract_attrs(submit_entities)

# 使用规则匹配得到实体关系三元组
schema = read_json(Path(DATA_DIR, 'schema.json'))
submit_relations = extract_relations(schema, submit_entities)


final_answer = {'attrs': submit_attrs,
                'entities': submit_entities,
                'relationships': submit_relations,
                }

with open('output/answers22.json', mode='w', encoding='UTF-8') as fw:
    json.dump(final_answer, fw, ensure_ascii=False, indent=4)

