# -*- coding: utf-8 -*-
import math
from tokenize import Double
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.nn import functional
from torch.nn import MSELoss
import torch.nn as nn
from tqdm import tqdm
import wandb
import random
from data_loader import *

random.seed(1)
#some parameters
_batch_size = 32
_lr = 1e-5
# Transformer Parameters
d_model = 646 # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 512  # dimension of K(=Q), V
n_layers = 3 # number of Encoder Layer
n_heads = 2  # number of heads in Multi-Head Attention


# wandb_config
# _config = {'train_data_ratio':_train_data_ratio,'barch_size' : _batch_size
#         ,'lr':_lr,'d_model':d_model,'d_ff':d_ff,'d_k':d_k,'n_layers':n_layers
#         ,'n_heads':n_heads
# }

#wandb init
# wandb.init(config=_config,
#                project='ash_rna',
#                dir='/user/hedongcheng/my_feature/Ash_Total_RNA_ii/transformer_encode',
#                job_type="training",
#                reinit=True)

rna_list_train_path = '/user/hedongcheng/my_feature/rna_list/train.txt'

with open(rna_list_train_path,'r') as f:
    rna_name_train = f.readlines()[0].split(' ') # the last char means subchain! 

rna_list_test_path = '/user/hedongcheng/my_feature/rna_list/test.txt'

with open(rna_list_test_path,'r') as f:
    rna_name_test = f.readlines()[0].split(' ') # the last char means subchain! 

random.shuffle(rna_name_train)
# extra_data
# rna_list_extra_train_path = '/user/hedongcheng/my_feature/Ash_Total_RNA_ii/RNA-FM/ash_seq_data/allseq_100.txt'
# with open(rna_list_extra_train_path,'r') as f:
#     rna_name_train += f.readlines()[0].split(' ') # the last char means subchain! 


# build dataloader
train_loader = Data.DataLoader(MyDataSet(rna_name_train), batch_size = _batch_size)
test_loader = Data.DataLoader(MyDataSet(rna_name_test), batch_size = _batch_size)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''

    batch_size, len_q = seq_q.size()[0:2]
    batch_size, len_k = seq_k.size()[0:2]
    # eq(zero) is PAD token
    pad_attn_mask = seq_k[:,:,0].data.eq(-1).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask # [batch_size, tgt_len, tgt_len]

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda(device=5)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda(device=5)(output + residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        #enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_inputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        
        return enc_outputs,enc_self_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda(device=5)
        # The swish activation function
        self.act = lambda x: x*torch.sigmoid(x)
        self.dense = nn.Linear(src_len*d_model, src_len*src_len)
        self.dense2 = nn.Linear(src_len*d_model,1)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        # tensor to store decoder outputs
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_output,enc_attn = self.encoder(enc_inputs)

       
        enc_outputs  = self.dense(enc_output.view(-1,src_len*d_model)).view(-1,src_len,src_len)
        enc_outputs = self.act(enc_outputs)
        
        return enc_outputs.view(-1,src_len,src_len)



model = Transformer().cuda(device=5)
optimizer = optim.Adam(model.parameters(), lr=_lr)
loss_fn =MSELoss()
pre_ls = 10000000
flag = 0 # if flag>=3 stop training
e_ls = 0
print("training process begin!")

for epoch in range(100):
    ls = []
    pbar = tqdm(train_loader)
    for enc_inputs, enc_outputs in train_loader:
        '''
        enc_inputs: [batch_size, src_len, embdim]
        enc_outputs: [batch_size, tgt_len, tgt_len]
        '''
       
        enc_inputs, enc_outputs = enc_inputs.cuda(device=5), enc_outputs.cuda(device=5)
        outputs = model(enc_inputs)

        # normalization
        enc_outputs = enc_outputs.to(torch.float32)
        enc_outputs[enc_outputs>30] = 30
        enc_outputs = enc_outputs/30
        enc_outputs[enc_outputs<0] = 0

        #loss backward
        loss = loss_fn(outputs, enc_outputs)
        pbar.set_description(f'Training epoch {epoch}: loss {e_ls}')
        pbar.set_postfix({'loss':loss.item()})
        pbar.update(1)
        ls.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    e_ls = sum(ls)/len(ls)
    # print(f'Training loss {epoch}: {ls}')
    # wandb.log({"loss":ls},step=epoch)

    # store model
    if e_ls < pre_ls:
        torch.save(model.state_dict(), 'test.pth')
        pre_ls=e_ls
        flag = 0
    else:
        flag += 1

    if flag >= 3:
        print(f'Training END! The best Training loss : {e_ls}')
        break



