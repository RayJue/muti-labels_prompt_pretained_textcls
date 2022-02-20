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
import numpy as np
from transformers import (
    Trainer,
    BertTokenizer,
    TrainingArguments,
    BertForMaskedLM,
    EarlyStoppingCallback
)


def compute_metrics(pred):
    labels = pred.label_ids[:, 3]
    preds = pred.predictions[:, 3].argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


def dataset_builder(x: List[str], y: List[str], tokenizer: BertTokenizer, max_len: int) -> Dataset:
    data_dict = {'text': x, 'label_text': y}
    result = Dataset.from_dict(data_dict)
    def preprocess_function(examples):
        text_token = tokenizer(examples['text'], padding=True,truncation=True, max_length=max_len)
        text_token['labels'] = np.array(tokenizer(examples['label_text'], padding=True,truncation=True, max_length=max_len)["input_ids"])
        return text_token
    result = result.map(preprocess_function, batched=True)
    return result
