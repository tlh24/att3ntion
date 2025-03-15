import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module): 

    def __init__(self, embedding_dim, num_heads, masked=False, dropout_rate=0):
        super(SelfAttention, self).__init__()

        assert embedding_dim % num_heads == 0
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim//num_heads
        self.masked = masked

        self.Wq = nn.Linear(embedding_dim, self.head_dim * num_heads, bias=False)
        self.Wk = nn.Linear(embedding_dim, self.head_dim * num_heads, bias=False)
        self.Wv = nn.Linear(embedding_dim, self.head_dim * num_heads, bias=False)
        self.Wo = nn.Linear(embedding_dim, self.head_dim * num_heads, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        device = x.device

        batch_size, seq_len, embedding_dim = x.shape
        
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)  
        V = V.permute(0, 2, 1, 3)  

        dot_product = torch.matmul(Q, torch.transpose(K, -2, -1))

        dot_product = dot_product/(math.sqrt(self.head_dim))

        if self.masked:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            mask = mask.to(device)
            mask = mask.unsqueeze(0).unsqueeze(0) #broadcast head and batch dim
            dot_product = dot_product.masked_fill(mask, float('-inf'))
        
        softmax = torch.nn.Softmax(dim=3)
        
        dot_product = softmax(dot_product)
        dot_product = self.dropout(dot_product) 

        attention_scores = torch.matmul(dot_product, V)
        attention_scores = attention_scores.permute(0, 2, 1, 3)
        attention_scores = attention_scores.reshape(batch_size, seq_len, embedding_dim)

        output = self.Wo(attention_scores)

        return output