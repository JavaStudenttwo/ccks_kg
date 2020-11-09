import os
from pathlib import Path
import tqdm
from utils.preprocess_data import Article
from pytorch_pretrained_bert import BertModel, BertTokenizer


DATA_DIR = '../data'  # 输入数据文件夹
YANBAO_DIR_PATH = str(Path(DATA_DIR, 'yanbao_txt__'))
PRETRAINED_BERT_MODEL_DIR = '../bert-base-chinese/'


tokenizer = BertTokenizer.from_pretrained(
    os.path.join(PRETRAINED_BERT_MODEL_DIR, 'vocab.txt')
)

yanbao_texts = []
for yanbao_file_path in Path(YANBAO_DIR_PATH).glob('*.txt'):
    with open(yanbao_file_path, encoding='utf-8') as f:
        yanbao_texts.append(f.read())


article_tokens = []
for article in tqdm.tqdm([Article(t) for t in yanbao_texts]):
    for para_text in article.para_texts:
        for sent in article.split_into_sentence(para_text):
            sent_tokens = list(sent)
            for i in sent_tokens:
                article_tokens.append(i)

dict_list = []
dict = tokenizer.vocab
for i in dict:
    dict_list.append(i)

notin_tokens = []
for i in article_tokens:
    if i not in dict_list and i not in notin_tokens:
        notin_tokens.append(i)

num = 0
for i in notin_tokens:
    num = num + 1
    print(i)
print(num)