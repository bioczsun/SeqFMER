from ast import mod
import math
from pyexpat import model
from threading import local
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ExpActivation(nn.Module):
    """
    Exponential activation function from Koo & Ploenzke, 2021 (PMID: 34322657)
    """
    def __init__(self):
        super(ExpActivation, self).__init__()

    def forward(self, x):
        return torch.exp(x)

class Unsqueeze(torch.nn.Module):
    """
    Unsqueeze for sequential models
    """
    def forward(self, x):
        return x.unsqueeze(-1)

class DeepSEA(nn.Module):
    def __init__(self,classes,linear_units,activate,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        conv_kernel_size = 8
        pool_kernel_size = 2
        if activate == 'relu':
            activation = nn.ReLU()
        elif activate == 'exp':
            activation = ExpActivation()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=320, kernel_size=conv_kernel_size, stride=1, padding=0),
            activation,
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Conv1d(in_channels=320, out_channels=480, kernel_size=conv_kernel_size, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Conv1d(in_channels=480, out_channels=960, kernel_size=conv_kernel_size, stride=1, padding=0),
            nn.BatchNorm1d(960),
            nn.ReLU(),
            nn.Dropout(0.2)
            )
        self.fc = nn.Sequential(
            nn.Linear(linear_units, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class Basset(nn.Module):
    def __init__(self, classes,linear_units,activate,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if activate == 'relu':
            activation = nn.ReLU()
        elif activate == 'exp':
            activation = ExpActivation()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=300, kernel_size=19, stride=1, padding=9),
            nn.BatchNorm1d(300),
            activation,
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=300, out_channels=200, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Conv1d(in_channels=200, out_channels=200, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=linear_units, out_features=1000),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1000, out_features=1000),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1000, out_features=classes)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class DanQ(nn.Module):
    def __init__(self,classes,linear_units,activate,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if activate == 'relu':
            activation = nn.ReLU()
        elif activate == 'exp':
            activation = ExpActivation()

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=4,out_channels=320,kernel_size=19,padding=9),
            activation,
            nn.MaxPool1d(kernel_size=13, stride=13),
            nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(input_size=320, hidden_size=320, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(linear_units, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, classes)
        )
    
    def forward(self, x):
        x = self.conv1d(x)
        x = x.permute(0,2,1)
        out,(hn,cn) = self.lstm(x)
        out = out.transpose(1,2)
        out = out.contiguous().view(x.size()[0],-1)
        out = self.fc(out)
        return out
    

class ExplaiNN(nn.Module):
    def __init__(self,classes, input_length,activate,num_cnns=300,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if activate == 'relu':
            activation = nn.ReLU()
        elif activate == 'exp':
            activation = ExpActivation()

        self.Conv1d = nn.Sequential(
                nn.Conv1d(in_channels=4, out_channels=num_cnns, kernel_size=19,padding=19//2),
                nn.BatchNorm1d(num_cnns),
                activation,
                nn.MaxPool1d(10),
                )
        self.Linear = nn.Sequential(
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(in_channels=int(input_length / 10)*num_cnns,
                          out_channels=100 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(100 * num_cnns),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(in_channels=100 * num_cnns,
                          out_channels=1 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(1 * num_cnns),
                nn.ReLU(),
                nn.Flatten(),
        )

        self.classifier = nn.Linear(num_cnns,classes)

    def forward(self,x):
        out = self.Conv1d(x)
        out = self.Linear(out)
        out1 = self.classifier(out)
        return out1
    
class PositionalEncoding(nn.Module):
    # Taken from: https://nlp.seas.harvard.edu/2018/04/03/attention.html
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
       
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
       
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)
    
class SATORI(nn.Module):
    def __init__(self,classes, linear_units,activate) -> None:
        super().__init__()
        self.numOutputChannels = 8*64
        self.numMultiHeads = 8
        self.SingleHeadSize = 64
        self.usePE = False

        if activate == 'relu':
            activation = nn.ReLU()
        elif activate == 'exp':
            activation = ExpActivation()

        if self.usePE:
            self.pe = PositionalEncoding(d_model = self.numOutputChannels,dropout=0.1)
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=self.numOutputChannels, kernel_size=19, padding=9),
            nn.BatchNorm1d(self.numOutputChannels),
            activation,
            nn.MaxPool1d(kernel_size=10),
            nn.Dropout(0.2)
        )
        self.RNN = nn.LSTM(input_size=self.numOutputChannels, hidden_size=self.numOutputChannels // 2, num_layers=2, batch_first=True, bidirectional=True,dropout=0.4)
        self.MultiheadAttention = nn.MultiheadAttention(embed_dim=self.numOutputChannels, num_heads=self.numMultiHeads)
        self.MultiheadLinear = nn.Sequential(nn.Linear(self.numOutputChannels,self.numOutputChannels),
                                             nn.ReLU())
        self.fc = nn.Linear(linear_units,classes)

    def forward(self,x):
        x = self.layer1(x)
        x = x.permute(0,2,1)
        x = self.RNN(x)
        x = x[0].permute(1,0,2)
        x = self.MultiheadAttention(x,x,x)
        x = x[0].permute(1,0,2)
        x = self.MultiheadLinear(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    

class CNN_Transformer(nn.Module):
    def __init__(self,classes,linear_units,activate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if activate == 'relu':
            activation = nn.ReLU()
        elif activate == 'exp':
            activation = ExpActivation()

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=300, kernel_size=19, padding=9),
            nn.BatchNorm1d(300),
            activation,
            nn.MaxPool1d(kernel_size=10),
            nn.Dropout(0.2)
        )
        self.transformer = nn.TransformerEncoderLayer(d_model=300,nhead=6)

        self.fc = nn.Sequential(
            nn.Linear(linear_units, 925),
            nn.ReLU(),
            nn.Linear(925, classes)
        )

    def forward(self, x):
        x = self.conv1d(x)
        x = x.permute(0,2,1)
        x = self.transformer(x)

        out = x.contiguous().view(x.size()[0],-1)
        out = self.fc(out)
        return out
    

class CNN_Attention(nn.Module):
    def __init__(self,clases,linear_units,activate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if activate == 'relu':
            activation = nn.ReLU()
        elif activate == 'exp':
            activation = ExpActivation()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=4,out_channels=300,kernel_size=19,padding=9),
            nn.BatchNorm1d(300),
            activation,
            nn.MaxPool1d(10),
        )
        self.multiatten = nn.MultiheadAttention(300,4,batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(linear_units,1000),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1000,clases),
        )
    def forward(self,x):
        x = self.conv(x)
        x = x.transpose(1,2)
        x,weights = self.multiatten(x,x,x)
        x = x.transpose(1,2).reshape(x.size(0),-1)
        x = self.linear(x)
        return x
    

class CNN(nn.Module):
    def __init__(self, classes, linear_units,activate) -> None:
        super(CNN, self).__init__()

        if activate == 'relu':
            activation = nn.ReLU()
        elif activate == 'exp':
            activation = ExpActivation()

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=300, kernel_size=19, padding=9),
            nn.BatchNorm1d(300),
            activation,
            nn.MaxPool1d(6, 6),
        )
        self.dense = nn.Sequential(
            nn.Linear(linear_units, 300),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(300, classes),
        )


    def forward(self, x):
        out = self.conv1d(x)
        out1 = out.contiguous().view(x.size()[0], -1)
        out = self.dense(out1)
        return out



class DanQ_ExplaiNN(nn.Module):
    def __init__(self,input_length,num_cnns,classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=num_cnns, kernel_size=19, padding=9),
            nn.BatchNorm1d(num_cnns),
            nn.ReLU(),
            nn.MaxPool1d(6)
        )
        self.lstm = nn.LSTM(input_size=num_cnns, hidden_size=num_cnns, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(num_cnns*2*int(input_length/6),256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, classes)
        )
        self.explainn = nn.Sequential(
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(in_channels=int(input_length / 6)*num_cnns,
                          out_channels=10 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(10 * num_cnns),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(in_channels=10 * num_cnns,
                          out_channels=1 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(1 * num_cnns),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(num_cnns,classes)
        )

    def forward(self, x):
        x1 = self.conv1d(x)
        x = x1.permute(0,2,1)
        out,(hn,cn) = self.lstm(x)
        out = out.transpose(1,2)
        out = out.contiguous().view(x.size()[0],-1)
        out = self.fc(out)

        explainn_out = self.explainn(x1)
        return out,explainn_out


class Basset_ExplaiNN_MultiClass(nn.Module):
    def __init__(self,input_length,num_cnns,classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=num_cnns, kernel_size=19, padding=9),
            nn.BatchNorm1d(num_cnns),
            ExpActivation(),
        )
        self.basset_conv = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=num_cnns, out_channels=200, kernel_size=11, padding=5),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=200, out_channels=200, kernel_size=7, padding=3),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(in_features=int(input_length/8) * 200, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=classes)
        )
        self.explainn = nn.Sequential(
                nn.MaxPool1d(6),
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(in_channels=int(input_length / 6)*num_cnns,
                          out_channels=100 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(100 * num_cnns),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(in_channels=100 * num_cnns,
                          out_channels=1 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(1 * num_cnns),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(num_cnns,classes)
        )
    def forward(self,x):
        conv1 = self.conv1d(x)
        basset_out = self.basset_conv(conv1)
        explainn_out = self.explainn(conv1)
        return basset_out,explainn_out

class Basset_ExplaiNN(nn.Module):
    def __init__(self,input_length,num_cnns,classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=num_cnns, kernel_size=19, padding=9),
            nn.BatchNorm1d(num_cnns),
            ExpActivation(),
        )
        self.basset_conv = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=num_cnns, out_channels=200, kernel_size=11, padding=5),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=200, out_channels=200, kernel_size=7, padding=3),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(in_features=int(input_length/8) * 200, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=classes)
        )
        self.explainn = nn.Sequential(
                nn.MaxPool1d(6),
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(in_channels=int(input_length / 6)*num_cnns,
                          out_channels=100 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(100 * num_cnns),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(in_channels=100 * num_cnns,
                          out_channels=1 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(1 * num_cnns),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(num_cnns,classes)
        )
    def forward(self,x):
        conv1 = self.conv1d(x)
        basset_out = self.basset_conv(conv1)
        explainn_out = self.explainn(conv1)
        return basset_out,explainn_out

import math

def get_sinusoidal_pos_encoding(n_position, d_model):
    """
    计算 Sinusoidal 位置编码
    
    参数:
        n_position: 序列长度
        d_model: 词向量维度

    返回值:
        形状为 (n_position, d_model) 的位置编码矩阵
    """
    position = torch.arange(n_position).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pos_encoding = torch.zeros(n_position, d_model)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    return pos_encoding

class Multi_ExplaiNN(nn.Module):
    def __init__(self,input_length,num_cnns,classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=num_cnns, kernel_size=19, padding=9),
            nn.BatchNorm1d(num_cnns),
            nn.ELU(),
            nn.MaxPool1d(6),
        )
        
        self.pe = get_sinusoidal_pos_encoding(int(input_length/6),num_cnns)
        self.pos_encoding = get_sinusoidal_pos_encoding(int(input_length/6), num_cnns)
        self.multihead = nn.MultiheadAttention(embed_dim=num_cnns, num_heads=4, dropout=0.1,batch_first=True)
        self.LN = nn.LayerNorm(num_cnns)
        self.linear = nn.Sequential(
            nn.Linear(int(input_length/6)*num_cnns,1000),
            nn.ReLU(),
            nn.Linear(1000,classes),
        )

        self.explainn = nn.Sequential(
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(in_channels=int(input_length / 6)*num_cnns,
                          out_channels=100 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(100 * num_cnns),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(in_channels=100 * num_cnns,
                          out_channels=1 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(1 * num_cnns),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(num_cnns,classes)
        )

    def forward(self,x):
        conv1 = self.conv1d(x)
        expanded_pos_encoding = self.pos_encoding.unsqueeze(0).transpose(1, 2).expand(x.shape[0], 128, int(600/6)).to(x.device)
        conv2 = conv1 + expanded_pos_encoding
        conv2 = conv2.permute(0,2,1)
        multi_out = self.multihead(conv2,conv2,conv2)
        multi_out = multi_out[0]
        multi_out = multi_out.permute(0,2,1)
        multi_out = multi_out.contiguous().view(x.size()[0],-1)
        multi_out = self.linear(multi_out)
        explainn_out = self.explainn(conv1)
        return multi_out,explainn_out

