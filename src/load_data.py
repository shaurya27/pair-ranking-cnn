import gensim
import numpy as np
import torch
from torch.utils.data import DataLoader
from constant import *

# Load gensim model
if PRE_TRAINED_EMBEDDING:
    gensim_model = gensim.models.Word2Vec.load(GENSIM_MODEL_PATH)

    # prepare mapping
    word2id = {}
    id2word = {}
    for i,j in enumerate(gensim_model.wv.index2word):
        word2id[j]=i+1
        id2word[i+1]=j
    word2id['<unk>']=len(word2id)
    id2word[len(id2word)] = '<unk>'

    pretrained_word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word2id), WORD_DIM))
    pretrained_word_embeds[0] = np.zeros((1,1))
    for i,j in enumerate(gensim_model.wv.syn0):
        pretrained_word_embeds[i+1] = j

## Data generator to read csv line by line
def data_generator(filepath):
    with open(filepath, 'r') as f:
        for i,line in enumerate(f):
            if i==0:
                continue
            yield line.strip('\n').split(',')
    return

class CustomDataset():

    def __init__(self,file_path,length,word2idx):
        self.file_path = file_path
        self.length = length
        self.word2idx = word2idx
        self.gen = data_generator(self.file_path)

    def __getitem__(self,index):
        try:
            text = self.gen.next()
        except StopIteration:
            self.gen = data_generator(self.file_path)
            text = self.gen.next()
        q1 = text[3].split()
        q2 = text[4].split()
        label = int(text[5])
        x1 = [self.word2idx.get(word) if self.word2idx.get(word) else self.word2idx['<unk>'] for word in q1]
        x2 = [self.word2idx.get(word) if self.word2idx.get(word) else self.word2idx['<unk>'] for word in q2]
        q1_word_len = len(q1)
        q2_word_len = len(q2)
        return {'q1': text[3],'q2': text[4],'labels':label,'q1_word_id':x1,'q2_word_id':x2,"q1_word_len": q1_word_len,"q2_word_len":q2_word_len}

    def __len__(self):
        return self.length

def collate_fn(batch):
    q1_max_word = [item['q1_word_len'] for item in batch]
    q2_max_word = [item['q2_word_len'] for item in batch]
    q1_max_len = max(q1_max_word)
    q2_max_len = max(q2_max_word)
    # for using 4 size kernel
    if q1_max_len <4:
        q1_max_len =4
    if q2_max_len <4:
        q2_max_len =4
    q1_word_data = np.zeros((len(batch),q1_max_len))
    q2_word_data = np.zeros((len(batch),q2_max_len))
    for i,item in enumerate(batch):
        q1_word_data[i,:len(item['q1_word_id'])] = item['q1_word_id']
        q2_word_data[i,:len(item['q2_word_id'])] = item['q2_word_id']
    target = [item['labels'] for item in batch]
    q1 =[item['q1'] for item in batch]
    q2 =[item['q2'] for item in batch]
    return torch.tensor(q1_word_data),torch.tensor(q2_word_data),torch.tensor(target),q1,q2

### Load datas
train_dataloader = DataLoader(CustomDataset(TRAIN_DATA_PATH,TRAIN_DATA_LENGTH,word2id),
                              batch_size=BATCH_SIZE,collate_fn = collate_fn,shuffle=False)
valid_dataloader = DataLoader(CustomDataset(VALID_DATA_PATH,VALID_DATA_LENGTH,word2id),
                              batch_size=BATCH_SIZE,collate_fn = collate_fn,shuffle=False)
test_dataloader = DataLoader(CustomDataset(TEST_DATA_PATH,TEST_DATA_LENGTH,word2id),
                              batch_size=BATCH_SIZE,collate_fn = collate_fn,shuffle=False)
