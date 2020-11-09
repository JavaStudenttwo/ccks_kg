import json
import numpy as np
import random
from random import choice
from tqdm import tqdm
import collections

global num

def get_nearest_start_position_(S1_):
    nearest_start_list = []
    current_distance_list = []
    S1 = []
    S1_ = np.array(S1_)
    for i in S1_:
        j = np.argmax(i, 1)
        S1.append(j)

    for start_pos_list in S1:
        nearest_start_pos = []
        current_start_pos = 0
        current_pos = []
        flag = False
        for i, start_label in enumerate(start_pos_list):
            if start_label > 0:
                current_start_pos = i
                flag = True
            nearest_start_pos.append(current_start_pos)
            if flag > 0:
                if i-current_start_pos > 10:
                    current_pos.append(499)
                else:
                    current_pos.append(i-current_start_pos)
            else:
                current_pos.append(499)
        # print(start_pos_list)
        # print(nearest_start_pos)
        # print(current_pos)
        # print('-----')
        nearest_start_list.append(nearest_start_pos)
        current_distance_list.append(current_pos)
    return nearest_start_list, current_distance_list


def get_nearest_start_position(S1):
    nearest_start_list = []
    current_distance_list = []
    for start_pos_list in S1:
        nearest_start_pos = []
        current_start_pos = 0
        current_pos = []
        flag = False
        for i, start_label in enumerate(start_pos_list):
            if start_label > 0:
                current_start_pos = i
                flag = True
            nearest_start_pos.append(current_start_pos)
            if flag > 0:
                if i-current_start_pos > 10:
                    current_pos.append(499)
                else:
                    current_pos.append(i-current_start_pos)
            else:
                current_pos.append(499)
        # print(start_pos_list)
        # print(nearest_start_pos)
        # print(current_pos)
        # print('-----')
        nearest_start_list.append(nearest_start_pos)
        current_distance_list.append(current_pos)
    return nearest_start_list, current_distance_list

def locate_entity(token_list, entity):
    try:
        for i, token in enumerate(token_list):
            if entity.startswith(token):
                len_ = len(token)
                j = i+1 ; joined_tokens = [token]
                while len_ < len(entity):
                    len_ += len(token_list[j])
                    joined_tokens.append(token_list[j])
                    j = j+1
                if ''.join(joined_tokens) == entity:
                    return i, len(joined_tokens)
    except Exception:
        # print(entity,token_list)
        pass
    return -1, -1

def seq_padding(X):
    L = [len(x) for x in X]
    # ML =  config.MAX_LEN #max(L)
    ML = max(L)
    return [x + [0] * (ML - len(x)) for x in X]

def seq_padding_(X):
    tmp = [0.0 for i in range(len(X[0][0]))]
    L = [len(x) for x in X]
    # ML =  config.MAX_LEN #max(L)
    ML = max(L)
    for x in X:
        for i in range(ML - len(x)):
            x.append(tmp)
    return X


def char_padding(X):
    L_S = [len(x) for x in X]
    ML_S = max(L_S)
    L = [[len(t) for t in s] for s in X]
    # ML =  config.MAX_LEN #max(L)
    ML = max([max(l) for l in L])
    if ML <= 15:
        return [[t + [0] * (15 - len(t)) for t in s]+[[1] * 15 for i in range(ML_S-len(s))] for s in X]
    else:
        return [[t + [0] * (15 - len(t)) if len(t) <=15 else t[:15] for t in s]+[[1] * 15 for i in range(ML_S-len(s))] for s in X]
def get_pos_tags(tokens, pos_tags, pos2id):
    pos_labels = [pos2id.get(flag,1) for flag in pos_tags]
    if len(pos_labels) != len(tokens):
        print(pos_labels)
        return False
    return pos_labels

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]



# {
# "sent": "半导体行情的风险是什么",
# "sent_tokens": ["半", "导", "体", "行", "情", "的", "风", "险", "是", "什", "么"],
# "sent_token_ids": [1288, 2193, 860, 6121, 2658, 4638, 7599, 7372, 3221, 784, 720],
# "entity_labels": [{"entity_type": "研报", "start_token_id": 0, "end_token_id": 10, "start_index": 0, "end_index": 10,
# "entity_tokens": ["半", "导", "体", "行", "情", "的", "风", "险", "是", "什", "么"],
# "entity_name": "半导体行情的风险是什么"}],
# "tags": ["B-Report", "I-Report", "I-Report", "I-Report", "I-Report", "I-Report", "I-Report", "I-Report", "I-Report", "I-Report", "I-Report"],
# "tag_ids": [11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]},

class DataLoader(object):
    def __init__(self, origin_data, subj_type2id, batch_size=64, evaluation=True):

        self.batch_size = batch_size
        self.subj_type2id = subj_type2id
        self.evaluation = evaluation

        data = self.preprocess(origin_data)
        if not self.evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.num_examples = len(data)
        self.data = [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]

    def preprocess(self, data):
        processed = []
        for index, d in enumerate(data):
            # if index > 800:
            #     break
            # d = self.data[i]
            tokens = d['sent_tokens']
            if len(tokens) > 180:
                continue
            items = d['entity_labels']

            if items:
                tokens_ids = d['sent_token_ids']
                tags_ids = d['tag_ids']

                processed += [(tokens_ids, tags_ids)]
        return processed

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 2
        lens = [len(x) for x in batch[0]]

        batch, orig_idx = sort_all(batch, lens)
        T = np.array(seq_padding(batch[0]))

        tags = np.array(seq_padding(batch[1]))

        return (T, tags, orig_idx)


if __name__ == '__main__':
    s = [[[1,3,4],[2,2,2,2,2]],[[1,3,4,6,6,6,6,6,6],[2,2,2,2,2,7,7,7]]]
    print(char_padding(s))


