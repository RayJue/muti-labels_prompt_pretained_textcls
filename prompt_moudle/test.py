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
label_2_id = {}
id_2_label = {}
source_data_path = "/home/rayjue/python_project/paper_project-main/prompt_moudle/data"
bert_name_path = "/home/rayjue/share/public_model/chinese_roberta_wwm_ext_pytorch"
output = '/home/rayjue/python_project/paper_project-main/prompt_moudle/output'
tokenizer = BertTokenizer.from_pretrained(bert_name_path)
model = BertForMaskedLM.from_pretrained(output)

data_dict = {
    "train": os.path.join(source_data_path, "train.csv"),
    "test": os.path.join(source_data_path, "test.csv")
}
df_test = pd.read_csv(data_dict["test"], encoding="utf-8")
pred = []
true = []
external_words = []
df_test.dropna(how="any", axis=0, inplace=True)
for index, row in tqdm(iterable=df_test.iterrows(), total=df_test.shape[0]):
    text = "酒店[MASK]，" + row["review"]
    tokenized_text = tokenizer.tokenize(text)
    if len(tokenized_text) > 512:
        tokenized_text = tokenized_text[:512]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)
    
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
    segments_tensors = torch.tensor([segments_ids]).to('cuda')
    
    masked_index = tokenized_text.index('[MASK]')
    
    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)
    predicted_index = torch.argmax(predictions[0][0][masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    # print(predicted_token+str(row["label"]))
    if predicted_token not in ["差", "好"]:
        external_words.append(predicted_token)
        predicted_token = "差"
    y_pred = label_2_id[predicted_token]
    pred.append(y_pred)
    true.append(row["label"])
precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='binary')
acc = accuracy_score(true, pred)
print({'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall})


good_words = ["美", "舒", "服", "豪", "华", "丽", "亮", "错", "大", "宜", "明"]
def get_label(words: List[str]) -> int:
    for key, val in label_2_id.items():
        if key in words:
            return val
    for word in words:
        if word in good_words:
            return 1
    return 0
