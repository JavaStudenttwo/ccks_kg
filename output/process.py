import json

fr = open('answers11.json', 'r', encoding='utf-8')

ins = json.load(fr)

sentdic_list = {}

sentdic_list['attrs'] = ins['attrs']
sentdic_list['entities'] = ins['entities']
list_ = []
for i in ins['relationships']:
    if len(list(i[2])) > 3:
        list_.append(i)

sentdic_list['relationships'] = list_

with open('test.json', 'w', encoding = 'utf-8') as file_obj:
    json.dump(sentdic_list, file_obj,ensure_ascii=False, indent=4)

print('保存成功')

