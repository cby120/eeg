from model import EEGNetReproduce
import torch
import torch.nn as nn
import os
from data_utils import EEGDataset,trim_last
from torch.utils.data import Dataset, RandomSampler, Subset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
#定义模型
model = EEGNetReproduce(n_channels=64,n_classes=5,input_window_size=100).to(device)#指定参数
#定义loss
loss_f = nn.CrossEntropyLoss()
#定义优化器

learning_rate = 0.01
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
save_path = ''
#lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer,factor=)

#load 数据集
#这里打乱可能需要重新匹配一下
train_dataset = EEGDataset(data_dir="./data", shuffle=False, train=True)
val_dataset = EEGDataset(data_dir="./data", shuffle=False, train=False)

train_dataloader = DataLoader(train_dataset, 10, collate_fn=trim_last)
val_dataloader = DataLoader(val_dataset, 10, collate_fn=trim_last)
#k 折检验 从data角度出发这里可以修正

def get_correct(pred,y):
    pred_y = torch.argmax(pred,dim=1)
    correct = torch.sum(pred_y == y)
    return correct


##########################################时序长度上难以确定长度，不妨直接假设是输入600*64，检测一下模型效果
max_epoch = 5
for epoch in range(max_epoch):
    model.train()
    train_total_loss = 0
    train_total_num = 0
    train_total_correct = 0
    for x,y in train_dataloader:
        x = x.to(device)
        y = y.long().to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_f(pred,y)
        loss.backward()
        train_total_loss += loss.item()
        train_total_num += len(x)#获得batch数目
        optimizer.step()
        train_total_correct += get_correct(pred,y)#获得一个计算正确率的函数
        print(train_total_loss/train_total_num,train_total_correct/train_total_num)
    val_total_num = 0
    val_total_correct = 0
    with torch.no_grad():
        model.eval()
        for x,y in val_dataloader:
            (x,y) = (x,y).to(device)
            pred = model(x)
            val_total_num += len(x)#获得batch数目
            val_total_correct += get_correct(pred,y)#获得一个计算正确率的函数
    #记录结果
    avg_train_acc = train_total_correct/train_total_num
    avg_train_loss = train_total_loss/train_total_num
    avg_val_acc = val_total_correct/val_total_num
    print('avg_train_loss:',avg_train_loss,
          'avg_train_acc:',avg_train_acc,
          'avg_val_acc:', avg_val_acc)
    epoch_save_path = os.path.join(save_path+str(epoch)+".bin")
    torch.save(model.state_dict(),epoch_save_path)




