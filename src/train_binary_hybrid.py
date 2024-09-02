import argparse
import re
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

set_random_seed(args.seed)
fasta_file = args.fasta

fasta = pysam.FastaFile(fasta_file)

class BinaryDataset(Dataset):
    def __init__(self, data,label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx],self.label[idx]
    
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
model = models.Basset_ExplaiNN(600,128,1)
model.to(device)


#定义训练参数
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=float(args.lr)
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=2,gamma=0.8)


model_save_path = "%s_%s_%s_%s_allneg"%(args.model,args.seed,args.seqlen,args.activate)
early_stopping = utils.EarlyStopping(save_path=args.outpath,model_name=model_save_path,patience=3)



#计算标签的权重
counter = collections.Counter(data_["train_label"])
counter = dict(counter)
weights = torch.tensor([int(counter[k]) for k in counter],dtype=torch.float) / len(data_["train_label"])
samples_weights = weights[data_["train_label"]]
sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights,len(samples_weights),replacement=True)


# Create a DataLoader for the training set

train_dataset = BinaryDataset(data_["train_data"],data_["train_label"])
train_loader = DataLoader(train_dataset, batch_size=int(args.batch),drop_last=True,collate_fn=custom_collate_fn,num_workers=4,sampler=sampler) 


# Create a DataLoader for the validation set
val_dataset = BinaryDataset(data_["test_data"],data_["test_label"])
val_loader = DataLoader(val_dataset, batch_size=int(args.batch),drop_last=True,shuffle=False,collate_fn=custom_collate_fn,num_workers=4)


#define train
def train(model,train_dataloader,test_dataloader,optimizer,criterion,epochs,early_stopping,scheduler,save_path,model_name,device):
    '''
    train model
    '''
    train_loss_ls = []
    test_loss_ls = []
    train_auc_ls = []
    test_auc_ls = []
    min_lr = 0.00001  # 设定最小学习率为0.0001
    # optimizer_explainn = torch.optim.Adam(list(model.explainn.parameters()) + list(model.conv1d.parameters()), lr=float(args.lr))
    # scheduler_explainn = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_explainn,step_size=2,gamma=0.8)
    for epoch in range(epochs):
        model.train()
        total_step = len(train_dataloader)
        running_loss = 0.0
        train_auc = 0.0
        for index,(data,label) in enumerate(train_dataloader):

            ##get single batch data
            train_data =data.float().to(device).transpose(1,2)
            train_label = label.to(device).view(-1,1)


            optimizer.zero_grad()
            # optimizer_explainn.zero_grad()

            basset_out,explainn_out = model(train_data)
            basset_out = F.sigmoid(basset_out)
            explainn_out = F.sigmoid(explainn_out)
            loss = criterion(basset_out,train_label.float())
            explainn_loss = criterion(explainn_out,train_label.float())
            loss_all = loss + explainn_loss
            #orth_loss = models.cal_orth_loss(model.Conv1d.weight)

            running_loss += loss.item()
            
            loss_all.backward()
            optimizer.step()

            # explainn_loss.backward()
            # optimizer_explainn.step()
            
            y_scores = basset_out.cpu().detach().numpy()
            fpr, tpr, _ = roc_curve(train_label.cpu(), y_scores)

            auroc = auc(fpr, tpr)
            train_auc += auroc

            if (index + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, auc: {:.4f}'
                        .format(epoch + 1, epochs, index + 1, total_step, loss.item(), auroc))
        epoch_loss = running_loss / (len(train_dataloader))
        epoch_auc = train_auc / (len(train_dataloader))
        train_loss_ls.append(epoch_loss)
        train_auc_ls.append(epoch_auc)



        print("------------eval------------------------")
        test_total_step = len(test_dataloader)
        test_runing_loss = 0.0
        test_true = []
        test_score = []
        test_pred = []
        with torch.no_grad():
            model.eval()
            for _,(data,label) in enumerate(test_dataloader):

                ##get single batch data
                test_data = data.float().to(device).transpose(1,2)
                test_label = label.to(device).view(-1,1)

                #forward
                basset_out,explainn_out = model(test_data)
                basset_out = F.sigmoid(basset_out)


                test_loss = criterion(basset_out,test_label.float())
                test_runing_loss += test_loss.item()

                #calculate auc and acc,recall
                y_scores = basset_out.cpu().detach().numpy()
                binary_label = np.where(basset_out.cpu().detach().numpy()>0.5,1,0)
                test_true.extend(test_label.cpu().numpy())
                test_pred.extend(binary_label)
                test_score.extend(y_scores)

        fpr, tpr, _ = roc_curve(test_true,test_score)
        eval_auroc = auc(fpr, tpr)

        scheduler.step()
        # lambda_lr_scheduler.step()

        epoch_test_loss = test_runing_loss / test_total_step
            # 确保学习率不低于最小学习率
        for param_group in optimizer.param_groups:
            if param_group['lr'] < min_lr:
                param_group['lr'] = min_lr
        current_lr = optimizer.param_groups[0]['lr']
        print('当前Epoch [{}/{}], lr: {} Train Loss: {:.4f},Train AUC: {:.4f}, Eval Loss: {:.4f},Eval AUC: {:.4f}'.format(epoch + 1, epochs,current_lr, epoch_loss,epoch_auc,epoch_test_loss,eval_auroc))

        print(classification_report(test_true,test_pred))

        test_loss_ls.append(epoch_test_loss)
        test_auc_ls.append(eval_auroc)

        #early stopping and step scheduler
        early_stopping(-eval_auroc,model)
        # scheduler_explainn.step()

        #达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            eval(model,test_dataloader,save_path,model_name,criterion,device)
            break #跳出迭代，结束训练

    return train_loss_ls,test_loss_ls,train_auc_ls,test_auc_ls   



def eval(model,test_dataloader,save_path,model_name,criterion,device):
    eval_model = model
    eval_model.load_state_dict(torch.load("%s/%s_best_network.pth"%(save_path,model_name)))
    eval_model.to(device)
    test_runing_loss = 0.0
    test_true = []
    test_score = []
    test_pred = []
    with torch.no_grad():
        eval_model.eval()
        for _,(data,label) in enumerate(test_dataloader):
            test_data =data.float().to(device).transpose(1,2)
            test_label = label.to(device).view(-1,1)
            basset_out,explainn_out = model(test_data)
            outputs = F.sigmoid(basset_out)
            test_loss = criterion(outputs,test_label.float())
            test_runing_loss += test_loss.item()
            y_scores = outputs.cpu().detach().numpy()

            binary_label = np.where(outputs.cpu().detach().numpy()>0.5,1,0)
            test_true.extend(test_label.cpu().numpy())
            test_pred.extend(binary_label)
            test_score.extend(y_scores)


        fpr, tpr, _ = roc_curve(test_true,test_score)
        eval_auroc = auc(fpr, tpr)

        precision, recall, thresholds = precision_recall_curve(test_true,test_score)
        auprc = auc(recall, precision)

        df = pd.DataFrame({"fpr":fpr,"tpr":tpr})
        df.to_csv("%s/eval_result_%s.csv"%(save_path,model_name))
        df1 = pd.DataFrame({"precision":precision,"recall":recall})
        df1.to_csv("%s/eval_pr_result_%s.csv"%(save_path,model_name))
        f = open("%s/classification_report_%s.txt"%(save_path,model_name),"w")
        f.write(classification_report(test_true,test_pred))
        f.write("\neval_auroc:%s"%eval_auroc)
        f.write("\neval_auprc:%s"%auprc)
        f.close()

#start train

train_loss_ls,test_loss_ls,train_auc_ls,test_auc_ls = train(model,train_loader,val_loader,optimizer,criterion,epochs,early_stopping,scheduler,args.outpath,model_save_path,device) 
np.save("%s/train_loss_%s.npy"%(args.outpath,model_save_path),train_loss_ls)
np.save("%s/test_loss_%s.npy"%(args.outpath,model_save_path),test_loss_ls)
np.save("%s/train_auc_%s.npy"%(args.outpath,model_save_path),train_auc_ls)
np.save("%s/test_auc_%s.npy"%(args.outpath,model_save_path),test_auc_ls)  



