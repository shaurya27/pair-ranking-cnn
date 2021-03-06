from model import *
from constant import *
from load_data import *
import numpy as np
import time
import tqdm
from torch.autograd import Variable
from sklearn.metrics import f1_score

if WORD_SIZE == None:
WORD_SIZE = len(word2id)

model = Model(WORD_SIZE,WORD_DIM,NUM_FILTERS,FILTER_SIZES,DROPOUT,HIDDEN_SIZE,pretrained_word_embeds)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion = torch.nn.NLLLoss(size_average=False)
#criterion = torch.nn.BCELoss(reduction='sum')

def train():
    model.train()
    corrects ,total_loss,total_bce_loss,_size,f_score = 0,0,0,0,0
    for q1, q2, label,a,b in tqdm.tqdm(train_dataloader, mininterval=1,
                                  desc='Train Processing', leave=False):
        label = label.type(torch.LongTensor)
        q1 = Variable(q1)
        q2 = Variable(q2)
        label = Variable(label)
        optimizer.zero_grad()
        pred = model(q1, q2)
        loss = criterion(pred, label)
        #loss = criterion(pred.view(label.size()), label.type(torch.FloatTensor))
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()
        f_score+=f1_score(label.data.numpy(),(torch.max(pred, 1)[1].view(label.size()).data).data.numpy())
        #corrects += ((pred.type(torch.FloatTensor)>0.5).view(label.size()).data == label.type(torch.ByteTensor).data).sum()
        _size += train_dataloader.batch_size
    num_batches = (train_dataloader.dataset.length // train_dataloader.batch_size)
    f_score = f_score/num_batches
    return total_loss /_size , corrects ,(float(corrects) / _size) * 100, _size,f_score

def evaluate():
    model.eval()
    corrects ,total_loss,total_bce_loss,_size,f_score = 0,0,0,0,0
    for q1, q2, label,_,_ in tqdm.tqdm(valid_dataloader, mininterval=1,
                                  desc='validation Processing', leave=False):
        label = label.type(torch.LongTensor)
        q1 = Variable(q1)
        q2 = Variable(q2)
        label = Variable(label)
        pred = model(q1, q2)
        loss = criterion(pred, label)
        #loss = criterion(pred.view(label.size()), label.type(torch.FloatTensor))
        total_loss += loss.data
        corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()
        f_score+=f1_score(label.data.numpy(),(torch.max(pred, 1)[1].view(label.size()).data).data.numpy())
        #corrects += ((pred.type(torch.FloatTensor)>0.5).view(label.size()).data == label.type(torch.ByteTensor).data).sum()
        _size += valid_dataloader.batch_size
    num_batches = (valid_dataloader.dataset.length // valid_dataloader.batch_size)
    f_score = f_score/num_batches
    return total_loss /_size , corrects ,(float(corrects) / _size) * 100, _size,f_score

def save(filename):
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'valid_loss': valid_loss,'valid_accuracy':valid_accuracy,
            'valid_fscore':valid_fscore}
    torch.save(state, filename)

training_loss = []
validation_loss = []
train_accuracy = []
valid_accuracy =[]
best_fscore = None
total_start_time = time.time()

try:
    print('-' * 90)
    for epoch in range(1, NUM_EPOCH + 1):
        epoch_start_time = time.time()
        train_loss, train_corrects, train_acc, train_size,train_fscore = train()
        scheduler.step()
        training_loss.append(train_loss * 1000.)
        train_accuracy.append(train_acc/100.)

        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f} | accuracy {:.4f}%({}/{} |  f1-score {:.4f}'.format(
            epoch, time.time() - epoch_start_time, train_loss, train_acc, train_corrects, train_size,train_fscore))

        valid_loss, valid_corrects, valid_acc, valid_size,valid_fscore = evaluate()

        validation_loss.append(valid_loss * 1000.)
        valid_accuracy.append(valid_acc / 100.)

        epoch_start_time = time.time()
        print('-' * 90)
        print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | accuracy {:.4f}%({}/{} | f1-score {:.4f}'.format(
            epoch, time.time() - epoch_start_time, valid_loss, valid_acc, valid_corrects, valid_size,valid_fscore))
        print('-' * 90)
        if not best_fscore or best_fscore < valid_fscore:
            best_fscore = valid_fscore
            save('../save/checkpoint_epoch_'+str(epoch)+'_valid_loss_'+str(valid_loss)
              +'_valid_acc_'+str(valid_acc)+'_'+'valid_fscoe'+str(valid_fscore)+'_'+'.pth.tar')
except KeyboardInterrupt:
    print("-" * 90)
    print("Exiting from training early | cost time: {:5.2f}min".format(
        (time.time() - total_start_time) / 60.0))
