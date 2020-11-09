import jieba.posseg as pseg

from utils.functions import *

# # 属性抽取
#
# 通过规则抽取属性
#
# - 研报时间
# - 研报评级
# - 文章时间

# In[ ]:


def find_article_time(yanbao_txt, entity):
    str_start_index = yanbao_txt.index(entity)
    str_end_index = str_start_index + len(entity)
    para_start_index = yanbao_txt.rindex('\n', 0, str_start_index)
    para_end_index = yanbao_txt.index('\n', str_end_index)

    para = yanbao_txt[para_start_index + 1: para_end_index].strip()
    if len(entity) > 5:
        ret = re.findall(r'(\d{4})\s*[年-]\s*(\d{1,2})\s*[月-]\s*(\d{1,2})\s*日?', para)
        if ret:
            year, month, day = ret[0]
            time = '{}/{}/{}'.format(year, month.lstrip(), day.lstrip())
            return time

    start_index = 0
    time = None
    min_gap = float('inf')
    for word, poseg in pseg.cut(para):
        if poseg in ['t', 'TIME'] and str_start_index <= start_index < str_end_index:
            gap = abs(start_index - (str_start_index + str_end_index) // 2)
            if gap < min_gap:
                min_gap = gap
                time = word
        start_index += len(word)
    return time


def find_yanbao_time(yanbao_txt, entity):
    paras = [para.strip() for para in yanbao_txt.split('\n') if para.strip()][:5]
    for para in paras:
        ret = re.findall(r'(\d{4})\s*[\./年-]\s*(\d{1,2})\s*[\./月-]\s*(\d{1,2})\s*日?', para)
        if ret:
            year, month, day = ret[0]
            time = '{}/{}/{}'.format(year, month.lstrip(), day.lstrip())
            return time
    return None


# In[ ]:


def extract_attrs(entities_json):
    train_attrs = read_json(Path(DATA_DIR, 'attrs.json'))['attrs']

    seen_pingjis = []
    for attr in train_attrs:
        if attr[1] == '评级':
            seen_pingjis.append(attr[2])
    article_entities = entities_json.get('文章', [])
    yanbao_entities = entities_json.get('研报', [])

    attrs_json = []
    for file_path in tqdm.tqdm(list(Path(DATA_DIR, 'yanbao_txt').glob('*.txt'))):
        yanbao_txt = '\n' + Path(file_path).open(encoding='UTF-8').read() + '\n'
        for entity in article_entities:
            if entity not in yanbao_txt:
                continue
            time = find_article_time(yanbao_txt, entity)
            if time:
                attrs_json.append([entity, '发布时间', time])

        yanbao_txt = '\n'.join(
            [para.strip() for para in yanbao_txt.split('\n') if
             len(para.strip()) != 0])
        for entity in yanbao_entities:
            if entity not in yanbao_txt:
                continue

            paras = yanbao_txt.split('\n')
            for para_id, para in enumerate(paras):
                if entity in para:
                    break

            paras = paras[: para_id + 5]
            for para in paras:
                for pingji in seen_pingjis:
                    if pingji in para:
                        if '上次' in para:
                            attrs_json.append([entity, '上次评级', pingji])
                            continue
                        elif '维持' in para:
                            attrs_json.append([entity, '上次评级', pingji])
                        attrs_json.append([entity, '评级', pingji])

            time = find_yanbao_time(yanbao_txt, entity)
            if time:
                attrs_json.append([entity, '发布时间', time])
    attrs_json = list(set(tuple(_) for _ in attrs_json) - set(tuple(_) for _ in train_attrs))

    return attrs_json

# In[ ]:

