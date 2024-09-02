# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class ExpActivation(nn.Module):
#     """
#     Exponential activation function from Koo & Ploenzke, 2021 (PMID: 34322657)
#     """
#     def __init__(self):
#         super(ExpActivation, self).__init__()

#     def forward(self, x):
#         return torch.exp(x)

# class Unsqueeze(torch.nn.Module):
#     """
#     Unsqueeze for sequential models
#     """
#     def forward(self, x):
#         return x.unsqueeze(-1)
    
# class Encoder(nn.Module):
#     def __init__(self, input_dim, embedding_dim, num_embeddings):
#         super(Encoder, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.num_embeddings = num_embeddings
#         num_cnns = 64
#         self.explainn = nn.Sequential(
#                 nn.Conv1d(in_channels=input_dim, out_channels=num_cnns, kernel_size=19, stride=1, padding=9),
#                 nn.Flatten(),
#                 Unsqueeze(),
#                 nn.Conv1d(in_channels=int(600 / 6)*num_cnns,
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
#                 nn.Flatten()
#         )

#     def forward(self, x):
#         x = self.explainn(x)
#         return x

# class Decoder(nn.Module):
#     def __init__(self, embedding_dim, output_dim):
#         super(Decoder, self).__init__()

#         self.deconv1 = nn.ConvTranspose1d(embedding_dim, 256, kernel_size=3, padding=1)
#         self.deconv2 = nn.ConvTranspose1d(256, 128, kernel_size=7, padding=3)
#         self.deconv3 = nn.ConvTranspose1d(128, 64, kernel_size=11, padding=5)
#         self.deconv4 = nn.ConvTranspose1d(64, output_dim, kernel_size=19, padding=9)

#     def forward(self, x):
#         x = F.relu(self.deconv1(x))
#         x = F.relu(self.deconv2(x))
#         x = F.relu(self.deconv3(x))
#         x = torch.sigmoid(self.deconv4(x))
#         return x

# class VectorQuantizer(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim, commitment_cost):
#         super(VectorQuantizer, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.num_embeddings = num_embeddings
#         self.commitment_cost = commitment_cost

#         self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
#         self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

#     def forward(self, x):
#         x = x.permute(0, 2, 1).contiguous()
#         input_shape = x.shape

#         flat_x = x.view(-1, self.embedding_dim)
#         distances = (torch.sum(flat_x**2, dim=1, keepdim=True) 
#                      + torch.sum(self.embeddings.weight**2, dim=1)
#                      - 2 * torch.matmul(flat_x, self.embeddings.weight.t()))

#         encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
#         encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=x.device)
#         encodings.scatter_(1, encoding_indices, 1)

#         quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)

#         e_latent_loss = F.mse_loss(quantized.detach(), x,)
#         q_latent_loss = F.mse_loss(quantized, x.detach())
#         loss = q_latent_loss + self.commitment_cost * e_latent_loss

#         quantized = x + (quantized - x).detach()
#         avg_probs = torch.mean(encodings, dim=0)
#         perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

#         return quantized.permute(0, 2, 1).contiguous(), loss, perplexity,encoding_indices

# class VQVAE(nn.Module):
#     def __init__(self, input_dim, embedding_dim, num_embeddings, commitment_cost):
#         super(VQVAE, self).__init__()
#         self.encoder = Encoder(input_dim, embedding_dim, num_embeddings)
#         self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
#         self.decoder = Decoder(embedding_dim, input_dim)

#     def forward(self, x):
#         z = self.encoder(x)
#         quantized, vq_loss, perplexity,encoding_indices = self.vector_quantizer(z)
#         print(encoding_indices)
#         x_recon = self.decoder(quantized)
#         recon_loss = F.mse_loss(x_recon, x)
#         loss = recon_loss + vq_loss
#         return x_recon, loss

# # # 示例用法
# input_dim = 4  # DNA序列的one-hot编码维度
# embedding_dim = 2  # 嵌入维度
# num_embeddings = 512  # 代码本的大小
# commitment_cost = 0.25  # 承诺损失权重

# # # 初始化模型
# model = VQVAE(input_dim, embedding_dim, num_embeddings, commitment_cost)

# a = torch.randn(1, 4, 600)  # 生成一个随机的DNA序列
# x_hat, loss = model(a)  # 计算重构序列和损失
# print(f'重构序列形状: {x_hat.shape}, 损失: {loss.item()}')


import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_embeddings):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        num_cnns = 64
        self.explainn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=num_cnns, kernel_size=19, stride=1, padding=9),
            nn.BatchNorm1d(num_cnns),
            nn.ReLU(),
            nn.MaxPool1d(6),
            nn.Conv1d(in_channels=num_cnns, out_channels=num_cnns*2, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(num_cnns*2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=num_cnns*2, out_channels=num_cnns*2, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(num_cnns*2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        x = self.explainn(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embedding_dim, output_dim, sequence_length):
        super(Decoder, self).__init__()
        self.sequence_length = sequence_length // 6 // 2 // 2
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.deconv = nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=2, stride=2)
        self.deconv1 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=6, stride=6)
        self.deconv3 = nn.ConvTranspose1d(64, 32, kernel_size=1)
        self.deconv4 = nn.ConvTranspose1d(32, output_dim, kernel_size=1)

    def forward(self, x):
        # print(x.shape)
        x = self.deconv(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        input_shape = x.shape

        flat_x = x.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True) 
                     + torch.sum(self.embeddings.weight**2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embeddings.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        # print(quantized)

        e_latent_loss = F.mse_loss(quantized.detach(), x,)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized.permute(0, 2, 1).contiguous(), loss, perplexity, encoding_indices

class VQVAE(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_embeddings, commitment_cost, sequence_length):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_dim, embedding_dim, num_embeddings)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, input_dim, sequence_length)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, perplexity, encoding_indices = self.vector_quantizer(z)
        x_recon = self.decoder(quantized)
        recon_loss = F.mse_loss(x_recon, x)
        loss = recon_loss + vq_loss
        return x_recon, loss

# 示例用法
input_dim = 4  # DNA序列的one-hot编码维度
embedding_dim = 128  # 嵌入维度
num_embeddings = 32  # 代码本的大小
commitment_cost = 0.25  # 承诺损失权重
sequence_length = 600  # 序列长度

# 初始化模型
model = VQVAE(input_dim, embedding_dim, num_embeddings, commitment_cost, sequence_length)

a = torch.randn(32, 4, 600)  # 生成一个随机的DNA序列()
x_hat, loss = model(a)  # 计算重构序列和损失
print(f'重构序列形状: {x_hat.shape}, 损失: {loss.item()}')
