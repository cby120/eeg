from model import EEGNetReproduce
import torch
import torch.nn as nn
import os
from data_utils import EEGDataset,trim_last,trim_symm
from torch.utils.data import Dataset, RandomSampler, Subset, DataLoader
from tqdm import tqdm
from transform import PersudoFT,PCATransform,ICATransform,CWT

def get_correct(pred,y):
    pred_y = torch.argmax(pred,dim=1)
    correct = torch.sum(pred_y == y)
    return correct

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Pytorch v0.4.0
    #定义模型

    model = EEGNetReproduce(n_channels=64*5,n_classes=2,input_window_size=600).to(device)#指定参数
    #from models.eegnet import EEGNetv4_class

    #model = EEGNetv4_class(in_chans=64,n_classes=5,input_window_samples=400).to(device)
    #定义loss
    loss_f = nn.CrossEntropyLoss()
    #定义优化器

    learning_rate = 0.008
    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9,weight_decay=0.01)
    save_path = ''
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.98)

    # load 数据集
    # #这里打乱可能需要重新匹配一下
    train_dataset = EEGDataset(data_dir="./data", tasks=[3,4,5,6,7,8,9,10,11,12,13,14],shuffle=False, train=True)
    val_dataset = EEGDataset(data_dir="./data",tasks=[3,4,5,6,7,8,9,10,11,12,13,14],shuffle=False, train=False)

    # q = CWT()
    # train_dataset = EEGDataset(data_dir="./data", tasks=[3,4,5,6,7,8,9,10,11,12,13,14],transforms = q,subjects= [x for x in range(1, 110) if x not in [88, 90, 92, 100, 104, 106]],shuffle=False, train=True)
    # val_dataset = EEGDataset(data_dir="./data",tasks=[3,4,5,6,7,8,9,10,11,12,13,14], transforms = q,subjects= [x for x in range(1, 110) if x not in [88, 90, 92, 100, 104, 106]],shuffle=False, train=False)


    # train_dataset = EEGDataset(data_dir="./data", tasks=[3,4,5,6,7,8,9,10,11,12,13,14],shuffle=False, train=True)
    # val_dataset = EEGDataset(data_dir="./data",tasks=[3,4,5,6,7,8,9,10,11,12,13,14], shuffle=False, train=False)



    train_dataloader = DataLoader(train_dataset, 64, collate_fn=trim_symm,shuffle=True,num_workers=4)
    val_dataloader = DataLoader(val_dataset, 64, collate_fn=trim_symm)
    #k 折检验 从data角度出发这里可以修正
    ##########################################时序长度上难以确定长度，不妨直接假设是输入600*64，检测一下模型效果
    max_epoch = 50
    for epoch in range(max_epoch):
        model.train()
        train_total_loss = 0
        train_total_num = 0
        train_total_correct = 0
        pbar = tqdm(train_dataloader)
        
        for x,y in pbar:
            x = x.float().to(device)
            y = (y-1).long().to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_f(pred,y)
            loss.backward()
            train_total_loss += loss.item()*len(x)
            train_total_num += len(x)#获得batch数目
            optimizer.step()
            #lr_scheduler.step()
            train_total_correct += get_correct(pred,y)#获得一个计算正确率的函数
            pbar.set_description(f"loss {train_total_loss/train_total_num:.3f} correct:{train_total_correct/train_total_num:.3f}")
            # print(train_total_loss/train_total_num,train_total_correct/train_total_num)
        val_total_num = 0
        val_total_correct = 0
        with torch.no_grad():
            model.eval()
            for x,y in val_dataloader:
                x = x.float().to(device)
                y = (y-1).long().to(device)
                pred = model(x)
                val_total_num += len(x)#获得batch数目
                val_total_correct += get_correct(pred,y)#获得一个计算正确率的函数
        #记录结果
        avg_train_acc = train_total_correct/train_total_num
        avg_train_loss = train_total_loss/train_total_num
        avg_val_acc = val_total_correct/val_total_num
        # print('avg_train_loss:',avg_train_loss,
        #     'avg_train_acc:',avg_train_acc,
        print('avg_val_acc:', avg_val_acc)
        epoch_save_path = os.path.join(save_path+str(epoch)+".bin")
        torch.save(model.state_dict(),epoch_save_path)
        lr_scheduler.step()

    #5 分类，用的原始数据，长度用600窗，从中间截断 c
    #5 分类，59.71%,
    #4 分类，58.66%,采用pca效果55.4%
    #2 分类，train-83%附件,采用ica 82
