import json
import os


def label(entities2label):
    filePath = 'yanbao'
    labelPath = 'label-7-12'
    filename = os.listdir(filePath)
    for i in filename:
        yanbao_label(entities2label, os.path.join(filePath, i), os.path.join(labelPath, i))


def yanbao_label(entities2label, yanbao_filename, label_filename):
    f = open(yanbao_filename, encoding='utf-8')
    f_w = open(label_filename, 'w+', encoding='utf-8')
    while 1:
        line = f.readline()
        if not line:
            break
        f_w.writelines('sentence:' + line)
        # 遍历句子将句中的所有实体找出来
        sent_entity = []
        for i in entities2label:
            if i in line:
                sent_entity.append(entities2label[i])
        entities = '   '.join(sent_entity)
        if entities:
            ent_str = 'entities: ' + entities + '\n'
            f_w.writelines(ent_str)
    f.close()
    f_w.close()


def readentitys():
    # 读取文件
    # 取得种子知识图谱中的实体
    with open('entities.json', encoding='utf-8') as f:
        entities_dict = json.load(f)
    # 取得新抽取出的实体
    with open('answers-7-12.json', encoding='utf-8') as f:
        data_dict = json.load(f)
    new_entities_dict = data_dict['entities']
    # 合并新旧文件中的所有实体
    for ent_type, ents in new_entities_dict.items():
        entities_dict[ent_type] = entities_dict[ent_type] + list(set(new_entities_dict[ent_type] + ents))
    # 将所有类型的实体加进一个list中
    entities2label = {}
    for ent_type, ents in entities_dict.items():
        for entity in entities_dict[ent_type]:
            entities2label[entity] = entity + '-' + ent_type
    return entities2label


if __name__ == "__main__":
    entities2label = readentitys()
    label(entities2label)