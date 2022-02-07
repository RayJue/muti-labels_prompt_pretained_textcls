# from distutils.command import config
from distutils.command.config import config
import imp
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer,AutoModel,AutoConfig



class bert_plus(nn.Module):
    def __init__(self,config):
        super(bert_plus, self).__init__()
        self.config = config
        if config.mid:
            self.bert = AutoModel.from_pretrained(config.pretrained_dir, output_hidden_states=True, return_dict=True)
        else:
            self.bert = AutoModel.from_pretrained(config.pretrained_dir)
        self.dropout = config.dropout
        self.config = config
        
        if config.model_name == 'cnn':
      
            self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.emb_size)) for k in config.filter_sizes])
            self.fc = nn.Linear(config.num_filters * len(config.filter_sizes),config.num_classes)

        elif config.model_name == 'lstm':
            
            self.lstm = nn.LSTM(config.emb_size, config.hid_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
            self.fc = nn.Linear(config.hid_size*2,config.num_classes)
        elif config.model_name == 's2s':
            pass
        else:
            if config.mid:
                self.fc = nn.Linear(config.emb_size*2,config.num_classes)
            else:
                self.fc = nn.Linear(config.emb_size,config.num_classes)
    
    def conv_part(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    def forward(self,input_ids,input_mask,token_type_ids):
        out = self.bert(input_ids=input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        
        if self.config.model_name == 'cnn':
            out = out.unsqueeze(1)
            out = torch.cat([self.conv_part(out, conv) for conv in self.convs], 1)
            out = self.dropout(out)
        elif self.config.model_name == 'lstm':
            out, _ = self.lstm(out)
            out = out[:, -1, :]
        else:
            if self.config.mid == True: 
                hidden_states = out.hidden_states 
                out = hidden_states[1][:, 0, :]  # [bs, hidden]
                for i in range(12, 13):
                    out = torch.cat((out.unsqueeze(1), hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
                cur_size = out.shape[0]
                out = out.reshape(cur_size,-1,)
            else:
                out = out['last_hidden_state']

        out = self.fc(out)

        return out


            
