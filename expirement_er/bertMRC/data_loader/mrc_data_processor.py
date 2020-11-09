#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


# Author: Xiaoy LI 
# Description:
# mrc_ner_data_processor.py


import os
from expirement_er.bertMRC.data_loader.mrc_utils import read_mrc_ner_examples
from expirement_er.bertMRC.data_loader.entities_type import *


class QueryNERProcessor(object):
    # processor for the query-based ner dataset 
    def get_train_examples(self, data_dir):
        data = read_mrc_ner_examples(os.path.join(data_dir, "train_.json"))
        return data

    def get_dev_examples(self, data_dir):
        return read_mrc_ner_examples(os.path.join(data_dir, "dev_.json"))

    def get_test_examples(self, data_dir):
        return read_mrc_ner_examples(os.path.join(data_dir, "dev_.json"))


class Baidu19Processor(QueryNERProcessor):
    def get_labels(self, ):
        return baidu10_type2id.keys()


class YanbaoProcessor(QueryNERProcessor):
    def get_labels(self, ):
        return ["NS", "NR", "NT", "O"]








