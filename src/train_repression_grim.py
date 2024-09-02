import argparse
import pysam
import random
import numpy as np
import pandas as pd

import time
import os


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


def pearson_loss(x,y):
        mx = torch.mean(x, dim=0, keepdim=True)
        my = torch.mean(y, dim=0, keepdim=True)
        xm, ym = x - mx, y - my
    
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        loss = torch.mean(1-cos(xm,ym))
        return loss

def pearson_r(x,y):
        mx = torch.mean(x, dim=0, keepdim=True)
        my = torch.mean(y, dim=0, keepdim=True)
        xm, ym = x - mx, y - my
    
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)


        return torch.mean(cos(xm,ym))

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
model = models.Basset_ExplaiNN(600,300,1)
model.to(device)


#定义训练参数
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    [param for name, param in model.named_parameters() if 'explainn' not in name],
    lr=float(args.lr),eps=0.000001
)
# optimizer = torch.optim.Adam(model.parameters(),lr=float(args.lr),eps=0.00001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=2,gamma=0.8)

model_name = "%s_%s_%s_%s_repression_mse"%(args.model,args.seed,args.seqlen,args.activate)
early_stopping = utils.EarlyStopping(save_path=args.outpath,model_name=model_name,patience=3)



# #计算标签的权重
# counter = collections.Counter(data_["train_label"])
# counter = dict(counter)
# weights = torch.tensor([int(counter[k]) for k in counter],dtype=torch.float) / len(data_["train_label"])
# samples_weights = weights[data_["train_label"]]
# sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights,len(samples_weights),replacement=True)


# Create a DataLoader for the training set

train_dataset = BinaryDataset(data_["train_data"],data_["train_label"])
train_loader = DataLoader(train_dataset, batch_size=int(args.batch),shuffle=True,collate_fn=custom_collate_fn,num_workers=8) 


# Create a DataLoader for the validation set
val_dataset = BinaryDataset(data_["test_data"],data_["test_label"])
val_loader = DataLoader(val_dataset, batch_size=int(args.batch),shuffle=False,collate_fn=custom_collate_fn,num_workers=8)

print("train_loader",len(train_loader))
print("val_loader",len(val_loader))





# Set the number of epochs and early stopping
# model_name = '%s_%s_%s_model_%s.pt'%(args.model,"mse",args.seqlen,args.seed)
num_epochs = args.epoch
# early_stopping = utils.EarlyStopping(save_path = args.outpath,model_name=model_name,patience=3,verbose=True, delta=0)


# Create a list to store the training loss
train_loss_list = []

# Create a list to store the validation loss
val_loss_list = []

# create a list to store the train time
time_train_list = []


# Train the model
optimizer_explainn = torch.optim.Adam(model.explainn.parameters(),lr=float(args.lr),eps=0.000001)
##calculate the time
start = time.time()
for epoch in range(num_epochs):
    train_loss = 0
    for index,(sequences, coverage) in enumerate(train_loader):
        sequences = sequences.to(device).permute(0,2,1)
        coverage = coverage.to(device).unsqueeze(1)
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        basset_out,explainn_out = model(sequences)
        basset_out = F.softplus(basset_out)
        explainn_out = F.softplus(explainn_out)
        loss = criterion(torch.log2(basset_out), torch.log2(coverage))
        explainn_loss = criterion(torch.log2(explainn_out), torch.log2(coverage))

        train_loss += loss.item()
        # Backward pass
        loss.backward(retain_graph=True)
        explainn_loss.backward()
        optimizer.step()
        optimizer_explainn.step()
        # Print the loss every 100 batches
        if index % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{index+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Pearson: {pearson_r(torch.log2(basset_out), torch.log2(coverage))}")
    # Compute the validation loss
    val_loss = 0
    val_pred = []
    val_true = []
    for sequences, coverage in val_loader:
        sequences = sequences.to(device).permute(0,2,1)
        coverage = coverage.to(device).unsqueeze(1)
        basset_out,explainn_out = model(sequences)
        basset_out = F.softplus(basset_out)
        explainn_out = F.softplus(explainn_out)
        val_pred.extend(basset_out.cpu().detach().numpy())
        val_true.extend(coverage.cpu().detach().numpy())
        loss = criterion(torch.log2(basset_out), torch.log2(coverage))# + 2 * pearson_loss(torch.log2(outputs), torch.log2(coverage))
        val_loss += loss.item()

    # Print the training and validation loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f},Pearson: {pearson_r(torch.log2(basset_out), torch.log2(coverage))}")
    # Append the training and validation loss to the respective lists
    train_loss_list.append(train_loss/len(train_loader))
    val_loss_list.append(val_loss/len(val_loader))
    # Check if the validation loss has improved
    early_stopping(val_loss/len(val_loader), model)
    if early_stopping.early_stop:
        print("Early stopping")
        #save val_pred and val_true npz
        # np.savez(os.path.join(args.outpath,"%s_val.npz"%model_name),val_pred=np.vstack(val_pred),val_true=np.vstack(val_true))
        break
    np.savez(os.path.join(args.outpath,"%s_val.npz"%model_name),val_pred=np.vstack(val_pred),val_true=np.vstack(val_true))
    # Adjust the learning rate
    scheduler.step()


end = time.time()
print(f"Training time: {end-start}s")


