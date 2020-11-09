from itertools import product

from utils.functions import *

# # 关系抽取
#
# - 对于研报实体，整个文档抽取特定类型(行业，机构，指标)的关系实体
# - 其他的实体仅考虑与其出现在同一句话中的其他实体组织成特定关系

# In[ ]:


def extract_relations(schema, entities_json):
    relation_by_rules = []
    relation_schema = schema['relationships']
    unique_s_o_types = []
    so_type_cnt = defaultdict(int)
    for s_type, p, o_type in schema['relationships']:
        so_type_cnt[(s_type, o_type)] += 1
    for (s_type, o_type), cnt in so_type_cnt.items():
        if cnt == 1 and s_type != o_type:
            unique_s_o_types.append((s_type, o_type))

    for path in tqdm.tqdm(list(Path(DATA_DIR, 'yanbao_txt').glob('*.txt'))):
        with open(path, encoding='UTF-8') as f:
            entity_dict_in_file = defaultdict(lambda: defaultdict(list))
            main_org = None
            for line_idx, line in enumerate(f.readlines()):
                for sent_idx, sent in enumerate(split_to_sents(line)):
                    for ent_type, ents in entities_json.items():
                        for ent in ents:
                            if ent in sent:
                                if ent_type == '机构' and len(line) - len(ent) < 3 or \
                                        re.findall('[\(（]\d+\.*[A-Z]*[\)）]', line):
                                    main_org = ent
                                else:
                                    if main_org and '客户' in sent:
                                        relation_by_rules.append([ent, '客户', main_org])
                                entity_dict_in_file[ent_type][
                                    ('test', ent)].append(
                                    [line_idx, sent_idx, sent,
                                     sent.find(ent)]
                                )

            for s_type, p, o_type in relation_schema:
                s_ents = entity_dict_in_file[s_type]
                o_ents = entity_dict_in_file[o_type]
                if o_type == '业务' and not '业务' in line:
                    continue
                if o_type == '行业' and not '行业' in line:
                    continue
                if o_type == '文章' and not ('《' in line or not '》' in line):
                    continue
                if s_ents and o_ents:
                    for (s_ent_src, s_ent), (o_ent_src, o_ent) in product(s_ents, o_ents):
                        if s_ent != o_ent:
                            s_occs = [tuple(_[:2]) for _ in
                                      s_ents[(s_ent_src, s_ent)]]
                            o_occs = [tuple(_[:2]) for _ in
                                      o_ents[(o_ent_src, o_ent)]]
                            intersection = set(s_occs) & set(o_occs)
                            if s_type == '研报' and s_ent_src == 'test':
                                relation_by_rules.append([s_ent, p, o_ent])
                                continue
                            if not intersection:
                                continue
                            if (s_type, o_type) in unique_s_o_types and s_ent_src == 'test':
                                relation_by_rules.append([s_ent, p, o_ent])

    train_relations = read_json(Path(DATA_DIR, 'relationships.json'))['relationships']
    result_relations_set = list(set(tuple(_) for _ in relation_by_rules) - set(tuple(_) for _ in train_relations))

    submit_relations = []
    for i in result_relations_set:
        if len(list(i[2])) > 3:
            submit_relations.append(i)

    return submit_relations

# In[ ]: