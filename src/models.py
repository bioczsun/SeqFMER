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

from xLSTM import mLSTM

class DanQmLSTM(nn.Module):
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
        # self.lstm = nn.LSTM(input_size=320, hidden_size=320, num_layers=2, batch_first=True, bidirectional=True)
        self.mlstm = mLSTM(input_size=320, hidden_size=320, num_layers=2)

        self.fc = nn.Sequential(
            nn.Linear(linear_units, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, classes)
        )
    
    def forward(self, x):
        x = self.conv1d(x)
        x = x.permute(0,2,1)
        out,_ = self.mlstm(x)
        out = out.transpose(1,2)
        out = out.contiguous().view(x.size()[0],-1)
        out = self.fc(out)
        return out


# a = torch.randn(32,4,600)
# model = DanQ(1,14720,"relu")
# print(model(a).shape)

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
# class Basset_ExplaiNN(nn.Module):
#     def __init__(self,input_length,num_cnns,classes, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.conv1d = nn.Sequential(
#             nn.Conv1d(in_channels=4, out_channels=num_cnns, kernel_size=19, padding=9),
#             nn.BatchNorm1d(num_cnns),
#             ExpActivation(),
#         )
#         self.basset_conv = nn.Sequential(
#             nn.MaxPool1d(2),
#             nn.Conv1d(in_channels=num_cnns, out_channels=200, kernel_size=11, padding=5),
#             nn.BatchNorm1d(200),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Dropout(0.5),
#             nn.Conv1d(in_channels=200, out_channels=200, kernel_size=7, padding=3),
#             nn.BatchNorm1d(200),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Dropout(0.5),
#             nn.Flatten(),
#             nn.Linear(in_features=int(input_length/8) * 200, out_features=256),
#             nn.ReLU(),
#             nn.Linear(in_features=256, out_features=128),
#             nn.ReLU(),
#             nn.Linear(in_features=128, out_features=classes)
#         )
#         self.explainn = nn.Sequential(
#                 nn.MaxPool1d(6),
#                 nn.Flatten(),
#                 Unsqueeze(),
#                 nn.Conv1d(in_channels=int(input_length / 6)*num_cnns,
#                           out_channels=100 * num_cnns, kernel_size=1,
#                           groups=num_cnns),
#                 nn.BatchNorm1d(100 * num_cnns),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Conv1d(in_channels=100 * num_cnns,
#                           out_channels=1 * num_cnns, kernel_size=1,
#                           groups=num_cnns),
#                 nn.BatchNorm1d(1 * num_cnns),
#                 nn.ReLU(),
#                 nn.Flatten(),
#                 nn.Linear(num_cnns,classes)
#         )

#         self.auxiliary = nn.Sequential(
#             nn.AdaptiveMaxPool1d(10),
#             nn.Flatten(),
#             nn.Linear(in_features=num_cnns * 10, out_features=32),
#             nn.ReLU(),
#             nn.Linear(in_features=32, out_features=classes)
#         )
#     def forward(self,x):
#         conv1 = self.conv1d(x)
#         basset_out = self.basset_conv(conv1)
#         explainn_out = self.explainn(conv1)
#         auxiliary_out = self.auxiliary(conv1)
#         return basset_out,explainn_out,auxiliary_out
    
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


# a = torch.randn(32,4,600)
# model = Test(600,64,1)
# print(model(a).shape)
        
# fft = FFTConv1d(in_channels=128,out_channels=128,kernel_size=601,padding=300)
# a = torch.randn(32,128,600)
# print(fft(a).shape)
# model = Basset(1,14600,"relu")
# print(model)
# class RelativePositionEncoding(nn.Module):
#     def __init__(self, max_length, d_model):
#         super(RelativePositionEncoding, self).__init__()
#         self.max_length = max_length
#         self.d_model = d_model

#         self.relative_position_embeddings = nn.Embedding(2 * max_length - 1, d_model)

#     def forward(self, length):
#         positions = torch.arange(length, dtype=torch.long, device=self.relative_position_embeddings.weight.device)
#         relative_positions_matrix = positions[:, None] - positions[None, :]
#         relative_positions_matrix += self.max_length - 1
#         return self.relative_position_embeddings(relative_positions_matrix)

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads, max_length):
#         super(MultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.d_model = d_model

#         self.qkv_linear = nn.Linear(d_model, 3 * d_model)
#         self.out_linear = nn.Linear(d_model, d_model)
#         self.relative_position_encoding = RelativePositionEncoding(max_length, d_model // num_heads)

#     def forward(self, x):
#         batch_size, seq_length, _ = x.size()
#         qkv = self.qkv_linear(x).view(batch_size, seq_length, 3, self.num_heads, self.d_model // self.num_heads)
#         q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

#         q = q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
#         k = k.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
#         v = v.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)

#         relative_positions = self.relative_position_encoding(seq_length)  # (seq_length, seq_length, head_dim)

#         scores = torch.einsum('bhqd, bhkd -> bhqk', q, k)
#         relative_scores = torch.einsum('bhqd, qkd -> bhqk', q, relative_positions)

#         scores += relative_scores
#         attn = F.softmax(scores, dim=-1)

#         context = torch.einsum('bhqk, bhvd -> bhqd', attn, v)
#         context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.d_model)
#         return self.out_linear(context)

# class TransformerEncoderLayerWithRelativePosition(nn.Module):
#     def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1, max_length=512):
#         super(TransformerEncoderLayerWithRelativePosition, self).__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads, max_length)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#         self.activation = F.relu

#     def forward(self, src):
#         src2 = self.self_attn(src)
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src
    
# class Basset_ExplaiNN(nn.Module):
#     def __init__(self, input_length, num_cnns, classes, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.conv1d = nn.Sequential(
#             nn.Conv1d(in_channels=4, out_channels=num_cnns, kernel_size=19, padding=9),
#             nn.BatchNorm1d(num_cnns),
#             nn.ReLU(),
#             nn.MaxPool1d(6),
#         )
#         self.pe = nn.Embedding(int(input_length/6), num_cnns)
#         self.transformer = TransformerEncoderLayerWithRelativePosition(d_model=num_cnns, num_heads=6, dim_feedforward=512)
#         self.classifier = nn.Sequential(
#                 nn.Linear(num_cnns * int(input_length/6), 256),
#                 nn.ReLU(),
#                 nn.Linear(256, classes)
#                 )
#         self.explainn = nn.Sequential(
#                 nn.MaxPool1d(6),
#                 nn.Flatten(),
#                 Unsqueeze(),
#                 nn.Conv1d(in_channels=int(input_length / 6 / 6)*num_cnns,
#                           out_channels=10 * num_cnns, kernel_size=1,
#                           groups=num_cnns),
#                 nn.BatchNorm1d(10 * num_cnns),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Conv1d(in_channels=10 * num_cnns,
#                           out_channels=1 * num_cnns, kernel_size=1,
#                           groups=num_cnns),
#                 nn.BatchNorm1d(1 * num_cnns),
#                 nn.ReLU(),
#                 nn.Flatten(),
#                 nn.Linear(num_cnns, classes)
#         )
#     def forward(self, x):
#         conv1 = self.conv1d(x)
#         transformer = self.transformer(conv1.permute(0,2,1))
#         transformer = self.transformer(conv1.permute(0,2,1)) + conv1.permute(0,2,1)
#         transformer = transformer.contiguous().view(x.size()[0], -1)
#         transformer_out = self.classifier(transformer)
#         explainn_out = self.explainn(conv1)
#         return transformer_out, explainn_out

# class AbsolutePositionEncoding(nn.Module):
#     def __init__(self, d_model, max_len=512):
#         super(AbsolutePositionEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=0.1)
        
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, d_model)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.d_model = d_model

#         self.qkv_linear = nn.Linear(d_model, 3 * d_model)
#         self.out_linear = nn.Linear(d_model, d_model)
#         self.attention_dropout = nn.Dropout(0.1)

#     def forward(self, x):
#         batch_size, seq_length, _ = x.size()
#         qkv = self.qkv_linear(x).view(batch_size, seq_length, 3, self.num_heads, self.d_model // self.num_heads)
#         q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

#         q = q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
#         k = k.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
#         v = v.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)

#         scores = torch.einsum('bhqd, bhkd -> bhqk', q, k) / math.sqrt(self.d_model // self.num_heads)
#         attn = F.softmax(scores, dim=-1)
#         attn = self.attention_dropout(attn)

#         context = torch.einsum('bhqk, bhvd -> bhqd', attn, v)
#         context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.d_model)
#         return self.out_linear(context)

# class TransformerEncoderLayerWithAbsolutePosition(nn.Module):
#     def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1, max_length=512):
#         super(TransformerEncoderLayerWithAbsolutePosition, self).__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads)
#         self.position_encoding = AbsolutePositionEncoding(d_model, max_length)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#         self.activation = F.relu

#     def forward(self, src):
#         src = self.position_encoding(src)
#         src2 = self.self_attn(src)
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src

# class Basset_ExplaiNN(nn.Module):
#     def __init__(self, input_length, num_cnns, classes, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.conv1d = nn.Sequential(
#             nn.Conv1d(in_channels=4, out_channels=num_cnns, kernel_size=19, padding=9),
#             nn.BatchNorm1d(num_cnns),
#             nn.ReLU(),
#             nn.MaxPool1d(6),
#         )
#         self.pe = nn.Embedding(int(input_length/6), num_cnns)
#         self.transformer = TransformerEncoderLayerWithAbsolutePosition(d_model=num_cnns, num_heads=6, dim_feedforward=512)
#         self.classifier = nn.Sequential(
#                 nn.Linear(num_cnns * int(input_length/6), 256),
#                 nn.ReLU(),
#                 nn.Linear(256, classes)
#                 )
#         self.explainn = nn.Sequential(
#                 nn.MaxPool1d(6),
#                 nn.Flatten(),
#                 Unsqueeze(),
#                 nn.Conv1d(in_channels=int(input_length / 6 / 6)*num_cnns,
#                           out_channels=10 * num_cnns, kernel_size=1,
#                           groups=num_cnns),
#                 nn.BatchNorm1d(10 * num_cnns),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Conv1d(in_channels=10 * num_cnns,
#                           out_channels=1 * num_cnns, kernel_size=1,
#                           groups=num_cnns),
#                 nn.BatchNorm1d(1 * num_cnns),
#                 nn.ReLU(),
#                 nn.Flatten(),
#                 nn.Linear(num_cnns, classes)
#         )
#     def forward(self, x):
#         conv1 = self.conv1d(x)
#         transformer = self.transformer(conv1.permute(0,2,1))
#         transformer = self.transformer(conv1.permute(0,2,1)) + conv1.permute(0,2,1)
#         transformer = transformer.contiguous().view(x.size()[0], -1)
#         transformer_out = self.classifier(transformer)
#         explainn_out = self.explainn(conv1)
#         return transformer_out, explainn_out
# a = torch.randn(32, 4, 600)
# model = Basset_ExplaiNN(600, 300, 1)
# print(model(a)[0].shape)
    


# a = torch.randn(32,4,600)
# model = SATORI(1,131520,"relu")

# print(model(a).shape)

# class GRIM_Basset_ExplaiNN(nn.Module):
#     def __init__(self,input_length,num_cnns,linear_units,classes, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         self.conv1d = nn.Sequential(
#             nn.Conv1d(in_channels=4, out_channels=num_cnns, kernel_size=19, padding=9),
#             nn.BatchNorm1d(num_cnns),
#             nn.ReLU(),
#             nn.MaxPool1d(6)
#         )

#         self.basset_conv = nn.Sequential(
#             nn.Conv1d(in_channels=num_cnns, out_channels=200, kernel_size=11, padding=5),
#             nn.BatchNorm1d(200),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             # nn.Dropout(0.3),
#             nn.Conv1d(in_channels=200, out_channels=200, kernel_size=7, padding=3),
#             nn.BatchNorm1d(200),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Flatten(),
#             nn.Linear(in_features=linear_units, out_features=256),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
#         )

#         self.basset_linear = nn.Sequential(
#             nn.Linear(in_features=256, out_features=128),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
#             nn.Linear(in_features=128, out_features=classes)
#         )

#         self.explainn = nn.Sequential(
#                 nn.Flatten(),
#                 Unsqueeze(),
#                 nn.Conv1d(in_channels=int(input_length / 6)*num_cnns,
#                           out_channels=10 * num_cnns, kernel_size=1,
#                           groups=num_cnns),
#                 nn.BatchNorm1d(10 * num_cnns),
#                 nn.ReLU(),
#                 nn.Dropout(0.3),
#                 nn.Conv1d(in_channels=10 * num_cnns,
#                           out_channels=1 * num_cnns, kernel_size=1,
#                           groups=num_cnns),
#                 nn.BatchNorm1d(1 * num_cnns),
#                 nn.ReLU(),
#                 nn.Flatten(),
#                 nn.Linear(num_cnns,classes)
#         )

#         self.Discriminator = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#         )

#     def forward(self,x,device):

#         generator = torch.Generator(device=device)
#         generator.manual_seed(0)  # 你可以根据需要设置随机种子  
#         shuffled_indices = torch.argsort(torch.rand(x.shape, generator=generator,device=device), dim=1)
#         shuffled_x = torch.gather(x, dim=1, index=shuffled_indices)

#         local_embedding = self.conv1d(x)
#         global_embedding = self.basset_conv(local_embedding)

#         local_embedding_shuffle = self.conv1d(shuffled_x)
#         global_embedding_shuffle = self.basset_conv(local_embedding_shuffle)

#         basset_out = self.basset_linear(global_embedding)
#         explainn_out = self.explainn(local_embedding)

#         local_encoding_expanded = local_embedding.unsqueeze(-1)
#         global_encoding_expanded = global_embedding.unsqueeze(1).unsqueeze(1)
#         outer_product_tensor = local_encoding_expanded * global_encoding_expanded
#         global_max_pooled = torch.max(outer_product_tensor.view(x.shape[0], 150 * 200, 256), dim=1)[0]
#         positive_scores = self.Discriminator(global_max_pooled)

#         global_embedding_shuffle = global_embedding_shuffle.unsqueeze(1).unsqueeze(1)
#         outer_product_tensor_shuffle = local_encoding_expanded * global_embedding_shuffle
#         global_max_pooled_shuffle = torch.max(outer_product_tensor_shuffle.view(x.shape[0], 150 * 200, 256), dim=1)[0]
#         negative_scores = self.Discriminator(global_max_pooled_shuffle)

#         #计算loss
#         positive_loss = F.softplus(-positive_scores).mean()
#         negative_loss = F.softplus(negative_scores).mean()

#         loss = positive_loss + negative_loss



#         return loss,basset_out,explainn_out
    
# class Expert(nn.Module):
#     def __init__(self,input_dim, output_dim, kernel_size, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.conv = nn.Sequential(
#             nn.Conv1d(in_channels=4,out_channels=32,kernel_size=19,padding=9),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(3),
#             nn.Conv1d(in_channels=32,out_channels=32,kernel_size=11,padding=5),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(in_channels=32,out_channels=32,kernel_size=7,padding=3),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Flatten(),
#             nn.Linear(32*50,128),
#             nn.ReLU(),
#             nn.Linear(128,64)
#         )
#     def forward(self,x):
#         return self.conv(x)

# class MoeModel(nn.Module):
#     def __init__(self, input_dim, output_dim, num_experts,top_k, kernel_size=3):
#         super(MoeModel, self).__init__()
#         self.top_k = top_k
#         self.experts = nn.ModuleList([Expert(input_dim, output_dim, kernel_size) for _ in range(num_experts)])
#         self.gating_network = nn.Sequential(
#             nn.Conv1d(in_channels=4, out_channels=128, kernel_size=kernel_size, padding=kernel_size//2),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.MaxPool1d(10),
#             nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, padding=3),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Flatten(),
#             nn.Linear(128*30, num_experts),
#         )

#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64*self.top_k, 128),
#             nn.ReLU(),
#             nn.Linear(128,1))

#     def forward(self, x):
#         # Calculate gating weights
#         gating_weights = self.gating_network(x)
#         top_k_values, top_k_indices = torch.topk(F.softmax(gating_weights, dim=1), self.top_k, dim=1)

#         # 初始化输出和权重
#         expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
#         expanded_outputs = expert_outputs.unsqueeze(1).expand(-1, self.top_k, -1, -1)
        
#         # 调整索引张量的维度并执行 gather 操作
#         expanded_indices = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, expert_outputs.shape[2])
#         selected_outputs = expanded_outputs.gather(2, expanded_indices)

#         # 加权求和
#         final_output = (selected_outputs * top_k_values.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)

#         out = self.fc(final_output)
#         return out
    

# class Expert(nn.Module):
#     def __init__(self,input_dim, output_dim, kernel_size, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.conv = nn.Sequential(
#             nn.Conv1d(in_channels=4,out_channels=1,kernel_size=19,padding=9),
#             nn.BatchNorm1d(1),
#             nn.ReLU(),
#             nn.MaxPool1d(10),
#             nn.Flatten(),
#             Unsqueeze(),
#             nn.Conv1d(in_channels=60,
#                         out_channels=10, kernel_size=1,
#                         groups=1),
#             nn.BatchNorm1d(10 * 1),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Conv1d(in_channels=10 * 1,
#                         out_channels=32 * 1, kernel_size=1,
#                         groups=1),
#             nn.BatchNorm1d(32 * 1),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
#     def forward(self,x):
#         return self.conv(x)

# class MoeModel(nn.Module):
#     def __init__(self, input_dim, output_dim, num_experts,top_k, kernel_size=3):
#         super(MoeModel, self).__init__()
#         self.top_k = top_k
#         self.experts = nn.ModuleList([Expert(input_dim, output_dim, kernel_size) for _ in range(num_experts)])
#         self.gating_network = nn.Sequential(
#             nn.Conv1d(in_channels=4, out_channels=16, kernel_size=kernel_size, padding=kernel_size//2),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.MaxPool1d(10),
#             nn.Conv1d(in_channels=16, out_channels=16, kernel_size=7, padding=3),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Flatten(),
#             nn.Linear(16*30, num_experts),
#         )

#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(32*self.top_k, 128),
#             nn.ReLU(),
#             nn.Linear(128,1))

#     def forward(self, x):
#         # Calculate gating weights
#         gating_weights = self.gating_network(x)
#         top_k_values, top_k_indices = torch.topk(F.softmax(gating_weights, dim=1), self.top_k, dim=1)

#         # 初始化输出和权重
#         expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
#         expanded_outputs = expert_outputs.unsqueeze(1).expand(-1, self.top_k, -1, -1)
        
#         # 调整索引张量的维度并执行 gather 操作
#         expanded_indices = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, expert_outputs.shape[2])
#         selected_outputs = expanded_outputs.gather(2, expanded_indices)

#         # 加权求和
#         final_output = (selected_outputs * top_k_values.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)

#         out = self.fc(final_output)
#         return out
    
# a = torch.randn(32,4,600)
# model = MoeModel(600,1,300,30)
# print(model(a).shape)