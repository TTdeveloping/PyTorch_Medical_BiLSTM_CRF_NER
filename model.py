import torch.nn as nn
import pickle
from data_utils import BatchManager, get_data_with_windows
import numpy as np
from initial_model import *
import torch
from config_model import *


def get_fea(batch):
    word = batch[0]
    word = torch.tensor(word).long()
    label = batch[1]
    label = torch.tensor(label).long()
    bound = batch[2]
    bound = torch.tensor(bound).long()
    flag = batch[3]
    flag = torch.tensor(flag).long()
    radical = batch[4]
    radical = torch.tensor(radical).long()
    pinyin = batch[5]
    pinyin = torch.tensor(pinyin).long()
    return word, label, bound, flag, radical, pinyin


class Model(nn.Module):
    def __init__(self, dict):
        super(Model, self).__init__()
        self.num_word = len(dict['word'][0])
        self.num_label = len(dict['label'][0])
        self.num_bound = len(dict['bound'][0])
        self.num_flag = len(dict['flag'][0])
        self.num_radical = len(dict['radical'][0])
        self.num_pinyin = len(dict['pinyin'][0])
        self.word_dim = 100
        self.bound_dim = 20
        self.flag_dim = 50
        self.radical_dim = 50
        self.pinyin_dim = 50
        W_n = self.num_word
        W_D = self.word_dim
        B_n = self.num_bound
        B_D = self.bound_dim
        F_n = self.num_flag
        F_D = self.flag_dim
        R_n = self.num_radical
        R_D = self.radical_dim
        P_n = self.num_pinyin
        P_D = self.pinyin_dim

        self.word_embed = nn.Embedding(W_n, W_D, padding_idx=0)
        init_embedding(self.word_embed.weight)
        self.bound_embed = nn.Embedding(B_n, B_D, padding_idx=0)
        init_embedding(self.bound_embed.weight)
        self.flag_embed = nn.Embedding(F_n, F_D, padding_idx=0)
        init_embedding(self.flag_embed.weight)
        self.radical_embed = nn.Embedding(R_n, R_D, padding_idx=0)
        init_embedding(self.radical_embed.weight)
        self.pinyin_embed = nn.Embedding(P_n, P_D, padding_idx=0)
        init_embedding(self.pinyin_embed.weight)
        self.dropout = nn.Dropout(dropout)
        D = W_D + B_D + F_D + R_D + P_D
        C = len(dict['label'][0])

        self.bilstm = nn.LSTM(input_size=D, hidden_size=lstm_hiddens, num_layers=lstm_layers,
                              bidirectional=True, batch_first=True, bias=True)

        self.linear = nn.Linear(in_features=lstm_hiddens * 2, out_features=C, bias=True)
        init_linear_weight_bias(self.linear)

    def forward(self, word, bound, flag, radical, pinyin):
        word_embed = self.word_embed(word)
        bound_embed = self.bound_embed(bound)
        flag_embed = self.flag_embed(flag)
        radical_embed = self.radical_embed(radical)
        pinyin_embed = self.pinyin_embed(pinyin)
        word_con_embed = torch.cat((word_embed, bound_embed, flag_embed, radical_embed, pinyin_embed), 2)
        x = self.dropout(word_con_embed)
        x, _ = self.bilstm(x)
        x = torch.tanh(x)
        logit = self.linear(x)
        return logit


