import argparse
import pysam
import random
import numpy as np
import pandas as pd
import collections


import models
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve


params = argparse.ArgumentParser(description='train model')
params.add_argument("--data",help="peaks file",required=True)
params.add_argument("--model",help="model name",default="Basset")
params.add_argument("--linear_units",help="linear units",required=True)
params.add_argument("--activate",help="activate loss",required=True)
params.add_argument("--seqlen",help="sequence length",required=True)
params.add_argument("--seed",help="random seed",default=40,type=int)
params.add_argument("--device",help="device cuda",default="cuda:0")
# params.add_argument("--name",help="model sort name",default="1")
params.add_argument("--fasta",help="fasta seq file",required=True)
params.add_argument("--epoch",help="epochs",default=50)

params.add_argument("--lr",help="learn rata",default=0.00001)
params.add_argument("--batch",help="batch size",default=256)
params.add_argument("--outpath",help="model name")
args = params.parse_args()



def set_random_seed(random_seed = 40):
    # set random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.seed = random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed) 
    torch.cuda.manual_seed_all(random_seed)

fasta_file = args.fasta

fasta = pysam.FastaFile(fasta_file)

class BinaryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0:3],np.array(self.data[idx][3:],dtype=np.float16)
    
# Create a custom collate function
def custom_collate_fn(batch):
    # 初始化序列和标签的列表
    sequences = []
    labels = []
    
    # 遍历batch中的每个(data, label)对
    for item in batch:
        # 假设 utils.onehot_seq 和 fasta.fetch 已经正确定义，并可以这样使用
        sequence = utils.onehot_seq(fasta.fetch(item[0][0], int(item[0][1]), int(item[0][2])).upper())
        sequences.append(sequence)
        labels.append(item[1])
    
    # 将序列和标签列表转换为张量
    sequences_tensor = torch.FloatTensor(np.array(sequences))
    labels_tensor = torch.FloatTensor(np.array(labels))
    
    return sequences_tensor, labels_tensor   



batch_size = int(args.batch)
epochs = int(args.epoch)
data_ = np.load(args.data)
device = args.device if torch.cuda.is_available() else "cpu"

# Create a model
model = models.Basset_ExplaiNN(600,300,int(args.linear_units))
model.to(device)


#定义训练参数
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=float(args.lr),eps=0.0000001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=2,gamma=0.5)

model_save_path = "%s_%s_%s"%("multicalss",args.model,args.seed)
early_stopping = utils.EarlyStopping(save_path=args.outpath,model_name=model_save_path,verbose=True)

#定义数据集
train_data = BinaryDataset(data_["train_data"])
test_data = BinaryDataset(data_["test_data"])

train_dataloader = DataLoader(train_data,batch_size=batch_size,collate_fn=custom_collate_fn,shuffle=True,num_workers=5)

test_dataloader = DataLoader(test_data,batch_size=batch_size,collate_fn=custom_collate_fn,drop_last=True,num_workers=5)



#define train
def train(model,train_dataloader,test_dataloader,optimizer,criterion,epochs,early_stopping,scheduler,save_path,model_name,device):
    '''
    train model
    '''

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_predictions = []
        train_true_labels = []
        
        # 训练
        for index,(data, label) in enumerate(train_dataloader):
            data = data.float().to(device).transpose(1,2)
            label = label.to(device)

            optimizer.zero_grad()
            outputs = torch.sigmoid(model(data)[0])
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            train_predictions.append(outputs.detach().cpu().numpy())
            train_true_labels.append(label.cpu().numpy())

            if index % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{index+1}/{len(train_dataloader)}], Train Loss: {loss.item():.4f}')
        
        # 计算训练集 AUC
        train_predictions = np.concatenate(train_predictions)
        train_true_labels = np.concatenate(train_true_labels)
        train_auc_scores = []
        for class_idx in range(train_predictions.shape[1]):
            fpr, tpr, _ = roc_curve(train_true_labels[:, class_idx], train_predictions[:, class_idx])
            train_auc_scores.append(auc(fpr, tpr))
        train_auc = np.mean(train_auc_scores)

        # 测试
        model.eval()
        total_test_loss = 0.0
        test_predictions = []
        test_true_labels = []
        with torch.no_grad():
            for data, label in test_dataloader:
                data = data.float().to(device).transpose(1,2)
                label = label.float().to(device)

                outputs = torch.sigmoid(model(data)[0])
                loss = criterion(outputs, label)
                total_test_loss += loss.item()

                test_predictions.append(outputs.cpu().numpy())
                test_true_labels.append(label.cpu().numpy())

        # 计算测试集 AUC
        test_predictions = np.concatenate(test_predictions)
        test_true_labels = np.concatenate(test_true_labels)
        test_auc_scores = []
        for class_idx in range(test_predictions.shape[1]):
            fpr, tpr, _ = roc_curve(test_true_labels[:, class_idx], test_predictions[:, class_idx])
            test_auc_scores.append(auc(fpr, tpr))
        test_auc = np.mean(test_auc_scores)

        # 输出结果和早停检测
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {total_train_loss:.4f}, Test Loss: {total_test_loss:.4f}, Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}')
        early_stopping(-test_auc, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step()

#start train
train(model,train_dataloader,test_dataloader,optimizer,criterion,epochs,early_stopping,scheduler,args.outpath,model_save_path,device) 
# np.save("%s/train_loss_%s.npy"%(args.outpath,model_save_path),train_loss_ls)
# np.save("%s/test_loss_%s.npy"%(args.outpath,model_save_path),test_loss_ls)

set_random_seed(40)