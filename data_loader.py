import torch
import numpy as np
from torch.nn import functional
import torch.utils.data as Data

src_len = 100 # enc_input max sequence length

def make_data(rna):
    #path to training data
    rna_seq_path = '/user/hedongcheng/my_feature/Ash_Total_RNA_ii/rna_seq/'
    rna_csv_path = '/user/hedongcheng/my_feature/Ash_Total_RNA_ii/rna_csv/'
    rna_emb_path = '/user/hedongcheng/my_feature/fea_npz/'

    # 'Y':5,'N':5,'M':5,'Z':5,'O':5,'g':2,'a':0,'u':3,'t':4,'c':1,'R':5,'K':5,'H':5,'W':5,'D':5,'F':5,'B':5,'V':5,'J':5,'L':5,'O':5,'P':5,'Q':5,'S':5,'x':5


    #rna_vocab
    rna_vocab = { 'A' : 0, 'C' : 1, 'G' : 2, 'U' : 3,'T':4,'X':5}

    # for seq
    try:
        with open(rna_seq_path+rna[0:4]+".seq",'r') as f:
            seq = f.readlines()[1].strip()
    except:
    # for seq
        with open('/user/hedongcheng/my_feature/Ash_Total_RNA_ii/seq/'+rna,'r') as f:
            seq = f.readlines()[1].strip()

    seq_idx = [rna_vocab[i] for i in seq]
  
    
    label = torch.tensor(seq_idx)  # label显示的是索引

    num_class = 6
    # get one hot embedding 
    label2onehot = functional.one_hot(label, num_classes=num_class)

    try:
        #for rna_embedding
        rna_emb = np.load(rna_emb_path+rna[0:4]+rna[4]+"_fea.npz",allow_pickle=True)
        rna_emb = torch.from_numpy(rna_emb['rna_fm'])
    except:
        #/user/hedongcheng/my_feature/Ash_Total_RNA_ii/outcsv
        rna_emb = np.load('/user/hedongcheng/my_feature/Ash_Total_RNA_ii/ash_4000_resuts/representations/'+rna.split('.')[0]+".npy",allow_pickle=True)
        rna_emb = torch.from_numpy(rna_emb)

    

    #concat onehot(seq) + rna embedding as encoder input
    enc_input = torch.cat((rna_emb,label2onehot),1)  # L*(5+embdim)

    try:
        #for contact map
        rna_csv= np.genfromtxt(rna_csv_path+rna[0:4]+'.csv', delimiter='\t')[:,:-1]
    except:
        rna_csv= np.genfromtxt('/user/hedongcheng/my_feature/Ash_Total_RNA_ii/outcsv/'+rna.split('.')[0]+'.csv', delimiter=' ')[:,:]

    enc_output = torch.from_numpy(rna_csv) # L*L
    #enc_input = rna_emb
    enc_output = torch.nn.functional.pad(enc_output,(0,src_len-enc_output.shape[0],0,src_len-enc_output.shape[0]),"constant",-1)
    enc_input = torch.nn.functional.pad(enc_input,(0,0,0,src_len-enc_input.shape[0]),"constant",-1)

    return enc_input, enc_output # [L,(5+embdim)], [L*L]


class MyDataSet(Data.Dataset):
  def __init__(self, rna_name):
    super(MyDataSet, self).__init__()
    self.rna_name = rna_name
  
  def __len__(self):
    return len(self.rna_name)
  
  def __getitem__(self, idx):
    rna = self.rna_name[idx]
    return make_data(rna) # enc_input(L*(5+embdim)), enc_output(L*L)