from operator import mod
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
import transformers
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import tqdm

from bert_muti import bert_plus
from config import Config
from dataloader import MyDataset
from utils import read_data,flat_accuracy


config = Config(model_name='')
model = bert_plus(config).to(config.device)

optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
loss_fn = nn.CrossEntropyLoss().to(config.device)

x_train, x_test, y_train, y_test, x_dev, y_dev = read_data(config)

train = Data.DataLoader(dataset=MyDataset(x_train, y_train), batch_size=config.batch_size, shuffle=True, num_workers=1)
test = Data.DataLoader(dataset=MyDataset(x_test, y_test), batch_size=config.batch_size, shuffle=True, num_workers=1)


def vaild(model,test):
    model.eval()
    
    with torch.no_grad():
        total_eval_accuracy = 0
        total_pred = []
        total_labels = []
        for i,batch in enumerate(test):
            x = tuple(item.to(config.device) for item in batch)
            pred = model(x[0], x[1], x[2])
            # pred = pred.data.max(dim=1, keepdim=True)[1]
            pred = pred.data.max(dim=1, keepdim=True)[1].squeeze().to('cpu').numpy().flatten()
            label_ids = batch[3].to('cpu').numpy().flatten()
            # total_eval_accuracy += flat_accuracy(pred, label_ids)
            total_pred.extend(pred)
            total_labels.extend(label_ids)

    return flat_accuracy(total_pred, total_labels)


# train

best_acc = 0
total_step = len(train)
for epoch in range(config.epoches):
    ep_pred = []
    ep_labels = []
    total_train_loss = 0
    total_eval_accuracy = 0
    total_iter = len(train)
    for i, batch in tqdm.tqdm(enumerate(train)):
        optimizer.zero_grad()
        batch = tuple(item.to(config.device) for item in batch)
        pred = model(batch[0], batch[1], batch[2])
        loss = loss_fn(pred.to(config.device), batch[3].to(config.device))
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('[{}|{}] step:{}/{} loss:{:.4f}'.format(epoch+1, config.epoches, i+1, total_step, loss.item()))
        acc = vaild(model,test)
        if acc>best_acc:
            torch.save(config.saved_dir+'/{}-{}.bin'.format(config.model_name if config.model_name else 'bert-resume',str(acc)[:4]))
            best_acc = acc
    print("Epoch: %d, Average training loss: %.4f"%(epoch, total_train_loss/total_step))
