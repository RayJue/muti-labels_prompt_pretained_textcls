import os
import pandas as pd
from tqdm import tqdm
import torch
import jieba
import codecs
import json
from sklearn.utils import shuffle
from typing import List, Dict
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import *
import numpy as np
from transformers import (
    Trainer,
    BertTokenizer,
    TrainingArguments,
    BertForMaskedLM,
    EarlyStoppingCallback
)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

source_data_path = "/home/rayjue/python_project/paper_project-main/prompt_moudle/data/data.csv"
bert_name_path = "/home/rayjue/share/public_model/chinese_roberta_wwm_ext_pytorch"

label_2_id = {}
id_2_label = {}


# f->few shot; z->zero_shot
z_or_f = "f"

tokenizer = BertTokenizer.from_pretrained(bert_name_path)
model = BertForMaskedLM.from_pretrained(bert_name_path)

from sklearn.model_selection import train_test_split
data_source_df = pd.read_csv(source_data_path)
data_source_df.dropna(how="any", axis=0, inplace=True)
train_df, test_df = train_test_split(data_source_df, test_size=0.2, random_state=42, shuffle=True)
print("Train contains [{}] records & Test contain s [{}] records".format(train_df.shape[0], test_df.shape[0]))

train_df.to_csv('/home/rayjue/python_project/paper_project-main/prompt_moudle/data/train.csv') 
test_df.to_csv('/home/rayjue/python_project/paper_project-main/prompt_moudle/data/test.csv') 
source_data_path = "/home/rayjue/python_project/paper_project-main/prompt_moudle/data"
data_dict = {
    "train": os.path.join(source_data_path, "train.csv"),
    "test": os.path.join(source_data_path, "test.csv")
}

df_train = pd.read_csv(data_dict["train"], encoding="utf-8")
df_test = pd.read_csv(data_dict["test"], encoding="utf-8")


text = []
label = []
punc = "＂!＃＄％＆＇?（）()/＊＋，－／：；,.＜＝＞＠［＼］\"＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。"
for index, row in tqdm(iterable=df_train.iterrows(), total=df_train.shape[0]):
    sentence = row["review"]
    words = jieba.lcut(sentence)
    for i in range(len(words)):
        sentence_train = "".join(words[:i])+"，酒店[MASK]，"+"".join(words[i:])
        sentence_test = "".join(words[:i])+"，酒店"+id_2_label[str(row["label"])]+"，"+"".join(words[i:])
        text.append(sentence_train)
        label.append(sentence_test)
text, label = shuffle(text, label)


eval_dataset = dataset_builder(text[:130], label[:130], tokenizer, 512)
train_dataset = dataset_builder(text[130:], label[130:], tokenizer, 512)


args = TrainingArguments(
    output_dir="/home/rayjue/python_project/paper_project-main/prompt_moudle/output",
    evaluation_strategy="steps",
    eval_steps=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-5,
    num_train_epochs=6,
    seed=20,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

