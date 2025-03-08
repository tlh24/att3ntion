import torch
import torch.nn as nn
import math

class HypergraphAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_rate=0):
        super(HypergraphAttention, self).__init__()
        
        assert embedding_dim % num_heads == 0
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wr = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Ws = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self.Wv_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv_r = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv_s = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self.Wo = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.shape
        
        Q = self.Wq(x)
        R = self.Wr(x)
        S = self.Ws(x)
        
        Vq = self.Wv_q(x)
        Vr = self.Wv_r(x)
        Vs = self.Wv_s(x)
        
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        R = R.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        S = S.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        Vq = Vq.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        Vr = Vr.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        Vs = Vs.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # compute 3-way attention scores of shape [b, h, i, j, k]
        dot_product = torch.einsum('bhid,bhjd,bhkd->bhijk', Q, R, S)
        dot_product = dot_product / (math.sqrt(self.head_dim))
        
        # Compute attention weights for each position
        # Aq - gathering to position i (softmax over j,k)
        dot_product_q = dot_product
        Aq = torch.softmax(dot_product_q.flatten(-2, -1), dim=-1).reshape(dot_product_q.shape)
        
        # Ar - gathering to position j (softmax over i,k)
        dot_product_r = dot_product.permute(0, 1, 3, 2, 4) 
        Ar = torch.softmax(dot_product_r.flatten(-2, -1), dim=-1).reshape(dot_product_r.shape)
        Ar = Ar.permute(0, 1, 3, 2, 4)  
        
        # As - gathering to position k (softmax over i,j)
        dot_product_s = dot_product.permute(0, 1, 4, 2, 3)  # [b, h, k, i, j]
        As = torch.softmax(dot_product_s.flatten(-2, -1), dim=-1).reshape(dot_product_s.shape)
        As = As.permute(0, 1, 3, 4, 2) 
        
        Aq = self.dropout(Aq)
        Ar = self.dropout(Ar)
        As = self.dropout(As)
        
        # Gather operations
        Y_q = torch.einsum('bhijk,bhjd,bhkd->bhid', Aq, Vr, Vs)  
        Y_r = torch.einsum('bhijk,bhid,bhkd->bhjd', Ar, Vq, Vs)  
        Y_s = torch.einsum('bhijk,bhid,bhjd->bhkd', As, Vq, Vr)  
        
        Y_combined = Y_q + Y_r + Y_s
        
        Y_combined = Y_combined.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embedding_dim)
        
        output = self.Wo(Y_combined)
        
        return output 