import re
from collections import defaultdict
from utils.preprocessing import *


DATA_DIR = './data'  # 输入数据文件夹


# ## 通过规则抽取实体
#
# - 机构
# - 研报
# - 文章
# - 风险

# In[ ]:


def aug_entities_by_rules(yanbao_dir):
    entities_by_rule = defaultdict(list)
    for file in list(yanbao_dir.glob('*.txt'))[:]:
        with open(file, encoding='utf-8') as f:
            found_yanbao = False
            found_fengxian = False
            for lidx, line in enumerate(f):
                # 公司的标题
                ret = re.findall('^[\(（]*[\d一二三四五六七八九十①②③④⑤]*[\)）\.\s]*(.*有限公司)$', line)
                if ret:
                    entities_by_rule['机构'].append(ret[0])

                # 研报
                if not found_yanbao and lidx <= 5 and len(line) > 10:
                    may_be_yanbao = line.strip()
                    if not re.findall(r'\d{4}\s*[年-]\s*\d{1,2}\s*[月-]\s*\d{1,2}\s*日?', may_be_yanbao) \
                            and not re.findall('^[\d一二三四五六七八九十]+\s*[\.、]\s*.*$', may_be_yanbao) \
                            and not re.findall('[\(（]\d+\.*[A-Z]*[\)）]', may_be_yanbao) \
                            and len(may_be_yanbao) > 5 \
                            and len(may_be_yanbao) < 100:
                        entities_by_rule['研报'].append(may_be_yanbao)
                        found_yanbao = True

                # 文章
                for sent in split_to_sents(line):
                    results = re.findall('《(.*?)》', sent)
                    for result in results:
                        entities_by_rule['文章'].append(result)

                # 风险
                for sent in split_to_sents(line):
                    if found_fengxian:
                        sent = sent.split('：')[0]
                        fengxian_entities = re.split('以及|、|，|；|。', sent)
                        fengxian_entities = [re.sub('^[■]+[\d一二三四五六七八九十①②③④⑤]+', '', ent) for ent in fengxian_entities]
                        fengxian_entities = [re.sub('^[\(（]*[\d一二三四五六七八九十①②③④⑤]+[\)）\.\s]+', '', ent) for ent in
                                             fengxian_entities]
                        fengxian_entities = [_ for _ in fengxian_entities if len(_) >= 4]
                        entities_by_rule['风险'] += fengxian_entities
                        found_fengxian = False
                    if not found_fengxian and re.findall('^\s*[\d一二三四五六七八九十]*\s*[\.、]*\s*风险提示[:：]*$', sent):
                        found_fengxian = True

                    results = re.findall('^\s*[\d一二三四五六七八九十]*\s*[\.、]*\s*风险提示[:：]*(.{5,})$', sent)
                    if results:
                        fengxian_entities = re.split('以及|、|，|；|。', results[0])
                        fengxian_entities = [re.sub('^[■]+[\d一二三四五六七八九十①②③④⑤]+', '', ent) for ent in fengxian_entities]
                        fengxian_entities = [re.sub('^[\(（]*[\d一二三四五六七八九十①②③④⑤]+[\)）\.\s]+', '', ent) for ent in
                                             fengxian_entities]
                        fengxian_entities = [_ for _ in fengxian_entities if len(_) >= 4]
                        entities_by_rule['风险'] += fengxian_entities

    for ent_type, ents in entities_by_rule.items():
        entities_by_rule[ent_type] = list(set(ents))
    return entities_by_rule