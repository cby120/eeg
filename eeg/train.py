from model import EEGNetReproduce
import torch
import torch.nn as nn
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Pytorch v0.4.0
#定义模型
model = EEGNetReproduce().to(device)#指定参数
#定义loss
loss_f = nn.CrossEntropyLoss()
#定义优化器

learning_rate = 0.01
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
save_path = ''
#lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer,factor=)

#load 数据集

dataset = data()
train_dataloader = 
val_dataloader = 
#need k 折检验 从data角度出发这里可以修正
def get_correct(pred,y):
    correct = 
    return correct

max_epoch = 5
for epoch in range(max_epoch):
    model.train()
    train_total_loss = 0
    train_total_num = 0
    train_total_correct = 0
    for x,y in train_dataloader:
        (x,y) = (x,y).to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_f(pred,y)
        loss.backward()
        train_total_loss += loss.item()
        train_total_num += len(x)#获得batch数目
        optimizer.step()
        train_total_correct += get_correct()#获得一个计算正确率的函数
    val_total_num = 0
    val_total_correct = 0
    with torch.no_grad():
        model.eval()
        for x,y in val_dataloader:
            (x,y) = (x,y).to(device)
            pred = model(x)
            val_total_num += len(x)#获得batch数目
            val_total_correct = get_correct()#获得一个计算正确率的函数
    #记录结果
    avg_train_acc = train_total_correct/train_total_num
    avg_train_loss = train_total_loss/train_total_num
    avg_val_acc = val_total_correct/val_total_num
    print('avg_train_loss:',avg_train_loss,
          'avg_train_acc:',avg_train_acc,
          'avg_val_acc:', avg_val_acc)
    epoch_save_path = os.path.join(save_path+str(epoch)+".bin")
    torch.save(model.state_dict(),epoch_save_path)




