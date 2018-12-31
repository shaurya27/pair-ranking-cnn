import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from constant import *

class Model(nn.Module):
    
    def __init__(self,word_size,word_dim,num_filters,filter_sizes,dropout,hidden_size,pretrained_word_embeds=None):
        super(Model,self).__init__()
        self.word_size = word_size
        self.word_dim = word_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.pretrained_word_embeds = pretrained_word_embeds
        self.embedding = nn.Embedding(self.word_size,self.word_dim,padding_idx=0)
        self.conv_left,self.conv_right = [],[]
        for i, filter_size in enumerate(self.filter_sizes):
            self.left = nn.Sequential(
                    nn.Conv2d(in_channels= 1, out_channels=self.num_filters, kernel_size=(filter_size, self.word_dim)),
                    nn.ReLU())
            self.right = nn.Sequential(
                    nn.Conv2d(in_channels= 1, out_channels=self.num_filters, kernel_size=(filter_size, self.word_dim)),
                    nn.ReLU())
            self.conv_left.append(self.left)
            self.conv_right.append(self.right)
        ins = len(self.filter_sizes) * self.num_filters
        self.simi_weight = nn.Parameter(torch.zeros(ins, ins))
        self.linear = nn.Linear(2*ins+1, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size,64)
        #self.out = nn.Linear(64, 2)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self._init_weights()
        
    def _init_weights(self):
        init.xavier_uniform_(self.simi_weight)
        if PRE_TRAINED_EMBEDDING:
            self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_word_embeds))
            if NON_TRAINABLE:
                self.embedding.weight.requires_grad = False
            else:
                self.embedding.weight.requires_grad = True
        else:
            init.xavier_uniform_(self.embedding.weight.data)

        
    def forward(self,leftx,rightx):
        leftx = leftx.type(torch.LongTensor)
        rightx = rightx.type(torch.LongTensor)
        enc_left = self.embedding(leftx)
        enc_right = self.embedding(rightx)
        enc_left = enc_left.unsqueeze(1)
        enc_right = enc_right.unsqueeze(1)
        enc_out_left, enc_out_right = [], []
        for (encoder_left, encoder_right) in zip(self.conv_left, self.conv_right):
            enc_left_ = encoder_left(enc_left)
            enc_right_ = encoder_left(enc_right)
            enc_left_ = F.max_pool2d(enc_left_,kernel_size=(enc_left_.size(2), 1))
            enc_right_ = F.max_pool2d(enc_right_,kernel_size=(enc_right_.size(2), 1))
            enc_left_ = enc_left_.squeeze(3)
            enc_left_ = enc_left_.squeeze(2)
            enc_right_ = enc_right_.squeeze(3)
            enc_right_ = enc_right_.squeeze(2)

            enc_out_left.append(enc_left_)
            enc_out_right.append(enc_right_)
            
        conc_left = torch.cat(enc_out_left, 1)
        conc_right = torch.cat(enc_out_right, 1)
        transform_left = torch.mm(conc_left, self.simi_weight)
        sims = torch.sum(torch.mm(transform_left,conc_left.t()), dim=1, keepdim=True)
        x = torch.cat([conc_left, sims, conc_right], 1)
        x = self.linear(x)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.out(x)
        #x_softmax = F.log_softmax(x)
        x_sigmoid = F.sigmoid(x)
        #return x_softmax
        return x_sigmoid