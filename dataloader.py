from distutils.command import config
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from config import Config
config = Config()

class MyDataset(Dataset):
  def __init__(self, sentences, labels=None, with_labels=True):
    self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_dir)
    self.with_labels = with_labels
    self.sentences = sentences
    self.labels = labels
    
  def __len__(self):
    return len(self.sentences)

  def __getitem__(self, index):
    # Selecting sentence1 and sentence2 at the specified index in the data frame
    sent = self.sentences[index]

    # Tokenize the pair of sentences to get token ids, attention masks and token type ids
    encoded_pair = self.tokenizer(sent,
                    padding='max_length',  # Pad to max_length
                    truncation=True,       # Truncate to max_length
                    max_length=config.pad_size,  
                    return_tensors='pt')  # Return torch.Tensor objects

    token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
    attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
    token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

    if self.with_labels:  # True if the dataset has labels
      label = self.labels[index]
      return token_ids, attn_masks, token_type_ids, label
    else:
      return token_ids, attn_masks, token_type_ids
