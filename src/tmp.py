import models
import torch



# a = torch.randn(4,4,2048)
model = models.CNN_Attention(16,18000,"relu")
# b = model(a)
# print(b.shape)

import torch
import torch.nn as nn

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

total_params, trainable_params = count_parameters(model)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")



# def calculate_nucleotide_probabilities(dna_sequence):
#     """
#     计算给定DNA序列中A、T、C、G四种核苷酸出现的概率
    
#     参数:
#     dna_sequence (str): 给定的DNA序列
    
#     返回:
#     dict: 一个字典,包含四种核苷酸及其出现概率
#     """
#     sequence_length = len(dna_sequence)
#     nucleotide_counts = {
#         'A': dna_sequence.count('A'),
#         'T': dna_sequence.count('T'),
#         'C': dna_sequence.count('C'),
#         'G': dna_sequence.count('G')
#     }
    
#     probabilities = {
#         nucleotide: count / sequence_length
#         for nucleotide, count in nucleotide_counts.items()
#     }
    
#     return probabilities

# # 使用示例
# dna_seq = "ATCGATCGACGATCATCGATCATCGATCATCG"
# nucleotide_probs = calculate_nucleotide_probabilities(dna_seq)
# print(list(nucleotide_probs.keys()))
# print(list(nucleotide_probs.values()))