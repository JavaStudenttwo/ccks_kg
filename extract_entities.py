from utils.functions import *
import random
import time
import os
from utils.preprocess_data import *
from collections import defaultdict
from utils.schemas import *
from utils.EvaluateScores import EvaluateScores
import torch
import tqdm
import logging
from model_entity.EtlModel import EntityModel


# - `extract_entities` 删除与训练集中重复的实体

# In[ ]:


def extract_entities(test_entities, train_entities):
    # test_entities = to_be_trained_entities
    # train_entities = read_json(Path(DATA_DIR, 'entities.json'))

    for ent_type, ents in test_entities.items():
        test_entities[ent_type] = list(set(ents) - set(train_entities[ent_type]))

    for ent_type in train_entities.keys():
        if ent_type not in test_entities:
            test_entities[ent_type] = []
    return test_entities


def evaluate(
        model, data_loader: torch.utils.data.DataLoader, logger: logging.Logger,
        tokenizer, device='cpu',
):
    founded_entities_json = defaultdict(set)
    golden_entities_json = defaultdict(set)
    for batch_id, one_data in enumerate(data_loader):
        T = torch.stack([d['T'] for d in one_data]).to(device)
        mask = torch.stack([d['mask'] for d in one_data]).to(device)
        subj_start_logits, subj_end_logits = model.predict_subj_per_instance(T, mask)

        for data, subj_start_list, subj_end_list in zip(one_data, subj_start_logits, subj_end_logits):

            for entity_label in data['entity_labels']:
                golden_entities_json[entity_label['entity_type']].add(entity_label['entity_name'])

            tokens = data['sent_tokens']
            for i, _ss1 in enumerate(subj_start_list):
                if _ss1 > 0:
                    for j, _ss2 in enumerate(subj_end_list[i:]):
                        if _ss2 == _ss1:
                            entity = ''.join(tokens[i: i + j + 1])
                            entity_type = entity_id2type[_ss2]
                            founded_entities_json[entity_type].add(entity)
                            break

    evaluate_tool = EvaluateScores(golden_entities_json, founded_entities_json)
    scores = evaluate_tool.compute_entities_score()
    return scores


def test(model, data_loader: torch.utils.data.DataLoader, logger: logging.Logger, device):
    founded_entities = defaultdict(set)
    for batch_id, one_data in enumerate(tqdm.tqdm(data_loader)):
        T = torch.stack([d['T'] for d in one_data]).to(device)
        mask = torch.stack([d['mask'] for d in one_data]).to(device)
        subj_start_logits, subj_end_logits = model.predict_subj_per_instance(T, mask)

        for data, subj_start_list, subj_end_list in zip(one_data, subj_start_logits, subj_end_logits):
            sent = data['sent']
            tokens = data['sent_tokens']
            for i, _ss1 in enumerate(subj_start_list):
                if _ss1 > 0:
                    for j, _ss2 in enumerate(subj_end_list[i:]):
                        if _ss2 == _ss1:
                            entity = ''.join(tokens[i: i + j + 1])
                            entity_type = entity_id2type[_ss2]
                            founded_entities[entity_type].add((entity, sent))
                            break
    result = defaultdict(list)
    for ent_type, ents in founded_entities.items():
        result[ent_type] = list(set(ents))
    return result


# ## 定义ner train loop， evaluate loop ，test loop

# In[ ]:


def train(model: EntityModel, data_loader: torch.utils.data.DataLoader, logger: logging.Logger, epoch_id,
          device='cpu'):
    pbar = tqdm.tqdm(data_loader)
    for batch_id, one_data in enumerate(pbar):

        T = torch.stack([d['T'] for d in one_data])
        S1 = torch.stack([d['S1'] for d in one_data])
        S2 = torch.stack([d['S2'] for d in one_data])
        mask = torch.stack([d['mask'] for d in one_data])

        loss = model.update(T, S1, S2, mask)
        pbar.set_description('epoch: {}, loss: {:.3f}'.format(epoch_id, loss))


# ## 整个训练流程是：
#
# - 使用数据集增强得到更多的实体
# - 使用增强过后的实体来指导训练
#
#
# - 训练后的模型重新对所有文档中进行预测，得到新的实体，加入到实体数据集中
# - 使用扩增后的实体数据集来进行二次训练，再得到新的实体，再增强实体数据集
# - (模型预测出来的数据需要`review_model_predict_entities`后处理形成提交格式)
#
#
# - 如果提交结果，需要`extract_entities`函数删除提交数据中那些出现在训练数据中的实体


# # ner主要训练流程
#
# - 分隔训练集验证集，并处理成dataset dataloader
# - 训练，验证，保存模型


def entity_train(logger, tokenizer, model, to_be_trained_entities, yanbao_texts):
    entities_json = to_be_trained_entities

    train_proportion = 0.9

    text_num = int(len(yanbao_texts))
    random.shuffle(yanbao_texts)
    yanbao_texts_train = yanbao_texts[:int(text_num * train_proportion)]
    yanbao_texts_dev = yanbao_texts[int(text_num * train_proportion):]

    train_preprocessed_datas = preprocess_data(entities_json, yanbao_texts_train, tokenizer)
    train_dataloader = build_dataloader(train_preprocessed_datas, tokenizer, batch_size=BATCH_SIZE)

    dev_preprocessed_datas = preprocess_data(entities_json, yanbao_texts_dev, tokenizer)
    dev_dataloader = build_dataloader(dev_preprocessed_datas, tokenizer, batch_size=BATCH_SIZE)

    best_evaluate_score = 0
    for epoch in range(TOTAL_EPOCH_NUMS):
        epoch_start_time = time.time()
        train(model, train_dataloader, logger=logger, epoch_id=epoch, device=DEVICE)
        # model.eval()
        evaluate_score = evaluate(model, dev_dataloader, logger=logger, tokenizer=tokenizer, device=DEVICE)
        f1 = evaluate_score['f']
        p = evaluate_score['p']
        r = evaluate_score['r']
        duration = time.time() - epoch_start_time
        print('f1：', f1, 'p：', p, 'r：', r, 'time：', duration)
        if f1 > best_evaluate_score:
            best_evaluate_score = f1
            save_model_path = os.path.join(SAVE_MODEL_DIR, 'best_en_model.pth')
            logger.info('saving model to {}'.format(save_model_path))
            model.save(save_model_path, epoch)