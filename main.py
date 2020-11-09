#!/usr/bin/env python
# coding: utf-8

# # CCKS 2020: 基于本体的金融知识图谱自动化构建技术评测
# 
# 竞赛背景
# 金融研报是各类金融研究结构对宏观经济、金融、行业、产业链以及公司的研究报告。报告通常是有专业人员撰写，对宏观、行业和公司的数据信息搜集全面、研究深入，质量高，内容可靠。报告内容往往包含产业、经济、金融、政策、社会等多领域的数据与知识，是构建行业知识图谱非常关键的数据来源。另一方面，由于研报本身所容纳的数据与知识涉及面广泛，专业知识众多，不同的研究结构和专业认识对相同的内容的表达方式也会略有差异。这些特点导致了从研报自动化构建知识图谱困难重重，解决这些问题则能够极大促进自动化构建知识图谱方面的技术进步。
#  
# 本评测任务参考 TAC KBP 中的 Cold Start 评测任务的方案，围绕金融研报知识图谱的自动化图谱构建所展开。评测从预定义图谱模式（Schema）和少量的种子知识图谱开始，从非结构化的文本数据中构建知识图谱。其中图谱模式包括 10 种实体类型，如机构、产品、业务、风险等；19 个实体间的关系，如(机构，生产销售，产品)、(机构，投资，机构)等；以及若干实体类型带有属性，如（机构，英文名）、（研报，评级）等。在给定图谱模式和种子知识图谱的条件下，评测内容为自动地从研报文本中抽取出符合图谱模式的实体、关系和属性值，实现金融知识图谱的自动化构建。所构建的图谱在大金融行业、监管部门、政府、行业研究机构和行业公司等应用非常广泛，如风险监测、智能投研、智能监管、智能风控等，具有巨大的学术价值和产业价值。
#  
# 评测本身不限制各参赛队伍使用的模型、算法和技术。希望各参赛队伍发挥聪明才智，构建各类无监督、弱监督、远程监督、半监督等系统，迭代的实现知识图谱的自动化构建，共同促进知识图谱技术的进步。
# 
# 竞赛任务
# 本评测任务参考 TAC KBP 中的 Cold Start 评测任务的方案，围绕金融研报知识图谱的自动化图谱构建所展开。评测从预定义图谱模式（Schema）和少量的种子知识图谱开始，从非结构化的文本数据中构建知识图谱。评测本身不限制各参赛队伍使用的模型、算法和技术。希望各参赛队伍发挥聪明才智，构建各类无监督、弱监督、远程监督、半监督等系统，迭代的实现知识图谱的自动化构建，共同促进知识图谱技术的进步。
# 
# 主办方邮箱  wangwenguang@datagrand.com kdd.wang@gmail.com
# 
# 
# 参考：https://www.biendata.com/competition/ccks_2020_5/
# 
# 平台GPU资源有限，建议本篇baseline参赛选手用于查看学习

# In[ ]:


import base64
from regulation import *
from extract_attrs import *
from extract_relations import *
from extract_entities import *



# ## 读入原始数据
# - 读入：所有研报内容
# - 读入：原始训练实体数据
yanbao_texts = []
for yanbao_file_path in Path(YANBAO_DIR_PATH).glob('*.txt'):
    with open(yanbao_file_path, encoding='utf-8') as f:
        yanbao_texts.append(f.read())

yanbao_texts = yanbao_texts[:20]
# 来做官方的实体训练集，后续会混合来自第三方工具，规则，训练数据来扩充模型训练数据
to_be_trained_entities = read_json(Path(DATA_DIR, 'entities.json'))


# 通过外部工具Hanlp来发现新实体
# 此任务十分慢, 但是只需要运行一次
tool_data_path = os.path.join('data', 'entities_by_third_party_tool.json')
if os.path.exists(tool_data_path):
    entities_by_third_party_tool = {}
    with open(tool_data_path, encoding='utf-8') as f:
        data = json.load(f)
        for key in data.keys():
            entities_by_third_party_tool[key] = data[key]
else:
    entities_by_third_party_tool = aug_entities_by_third_party_tool()
    with open(tool_data_path, 'w', encoding='utf-8') as file_obj:
        json.dump(entities_by_third_party_tool, file_obj, ensure_ascii=False)



# 将外部工具找到的新实体加入进现有的实体列表中
for ent_type, ents in entities_by_third_party_tool.items():
    to_be_trained_entities[ent_type] = list(set(to_be_trained_entities[ent_type] + ents))
# for k, v in entities_by_third_party_tool.items():
#     print(k)
#     print(set(v))


# 通过规则来寻找新的实体
entities_by_rule = aug_entities_by_rules(Path(DATA_DIR, 'yanbao_txt'))

# 将通过规则匹配找到的新实体加入进实体列表中
for ent_type, ents in entities_by_rule.items():
    to_be_trained_entities[ent_type] = list(set(to_be_trained_entities[ent_type] + ents))
# for k, v in entities_by_rule.items():
#     print(k)
#     print(set(v))


# ## 准备训练ner模型

# In[ ]:

logger = logging.getLogger(__name__)

UNCASED = './bert-base-chinese'  # your path for model and vocab
VOCAB = 'vocab.txt'
tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED,VOCAB))
bert_model = BertModel.from_pretrained(UNCASED)
bert_model.eval()
bert_model.cuda()

# 定义命名实体识别模型
model = EntityModel(opt, bert_model)

save_model_path = os.path.join(SAVE_MODEL_DIR, 'finnal_ccks_model.pth')
if Path(save_model_path).exists():
    model_state_dict = torch.load(save_model_path, map_location='cpu')
    model.load_state_dict(model_state_dict)


# 循环轮次数目
nums_round = opt['nums_round']
for i in range(nums_round):
    # 为避免自己看见自己的问题，对研报文章进行打散，将测试集和训练验证集分开
    text_num = int(len(yanbao_texts) / 2)
    random.shuffle(yanbao_texts)
    yanbao_texts_train = yanbao_texts[:text_num]
    yanbao_texts_test = yanbao_texts[text_num:]

    # train
    entity_train(logger, tokenizer, model, to_be_trained_entities, yanbao_texts_train)

    # 训练数据在main_train函数中处理并生成dataset dataloader，此处无需生成
    # 测试数据在此处处理并生成dataset dataloader
    test_preprocessed_datas = preprocess_data(to_be_trained_entities, yanbao_texts_test, tokenizer, for_train=False)
    test_dataloader = build_dataloader(test_preprocessed_datas, tokenizer, batch_size=BATCH_SIZE)

    # 读取train过程中保存下来的最佳模型
    save_model_path = os.path.join(SAVE_MODEL_DIR, 'best_en_model.pth')
    print("Loading model from {}".format(save_model_path))
    model.load(save_model_path)

    # 使用模型预测结果
    model_predict_entities = test(model, test_dataloader, logger=logger, device=DEVICE)
    # 前三轮模型效果不够好，不做预测
    if i in [0, 1, 2]:
        model_predict_entities = {}
    # 修复训练预测结果
    reviewed_entities = review_model_predict_entities(model_predict_entities)
    # 创造出提交结果，将种子知识图谱中已有的实体删除
    new_entities = extract_entities(reviewed_entities, to_be_trained_entities)

    # 将训练预测结果再次放入训练集中， 重新训练或者直接出结果
    for ent_type, ents in new_entities.items():
        to_be_trained_entities[ent_type] = list(set(to_be_trained_entities[ent_type] + ents))


# 创造出提交结果，将种子知识图谱中已有的实体删除
origin_trained_entities = read_json(Path(DATA_DIR, 'entities.json'))
submit_entities = extract_entities(to_be_trained_entities, origin_trained_entities)

# 使用规则匹配得到实体属性
train_attrs = read_json(Path(DATA_DIR, 'attrs.json'))['attrs']
submit_attrs = extract_attrs(submit_entities)

# 使用规则匹配得到实体关系三元组
schema = read_json(Path(DATA_DIR, 'schema.json'))
submit_relations = extract_relations(schema, submit_entities)


# ## 生成提交文件
#
# 根据biendata的要求生成提交文件
#
# 参考：https://www.biendata.com/competition/ccks_2020_5/make-submission/


final_answer = {'attrs': submit_attrs,
                'entities': submit_entities,
                'relationships': submit_relations,
                }

with open('output/answers.json', mode='w', encoding='UTF-8') as fw:
    json.dump(final_answer, fw, ensure_ascii=False, indent=4)

# with open('output/answers.json', 'rb') as fb:
#     data = fb.read()
#
# b64 = base64.b64encode(data)
# payload = b64.decode()
# html = '<a download="{filename}" href="data:text/json;base64,{payload}" target="_blank">{title}</a>'
# html = html.format(payload=payload,title='answers22.json',filename='answers22.json')
# HTML(html)
