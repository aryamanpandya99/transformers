import torch
import math
from torch import nn

def scaled_dot_product_attention(q, k, d_k, mask):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    return nn.Softmax(-1)(scores)

def positional_embedding(input_tensor: torch.Tensor, output_dim: int, n=10000):
    """
    Naive sin/cosin positional embedding
    """
    p = torch.zeros((input_tensor.shape[-1], output_dim))
    indices = torch.arange(input_tensor.size(-1))
    i_values = torch.arange(int(output_dim/2))
    denominators = torch.float_power(n, 2*i_values/output_dim)
    p[:, 0::2] = torch.sin(indices.unsqueeze(1) / denominators.unsqueeze(0))
    p[:, 1::2] = torch.cos(indices.unsqueeze(1) / denominators.unsqueeze(0))
    return p


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_model, d_v, dropout, num_heads, mask) -> None:
        super(MultiHeadAttention, self).__init__()
        self.d_k, self.d_v, self.d_model, self.num_heads = d_k, d_v, d_model, num_heads
        self.query_layer, self.key_layer, self.value_layer = nn.Linear(d_model, num_heads* d_k), nn.Linear(d_model, num_heads* d_k), nn.Linear(d_model, num_heads*d_v)
        self.layer_norm = nn.LayerNorm(d_model)
        self.concat_projection = nn.Linear(num_heads*d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.mask = mask

    def forward(self, q, k, v):
        k_len, q_len, v_len, batch_size = k.size(1), q.size(1), v.size(1),  q.size(0)
        residual = q
        k = self.key_layer(k).view(batch_size, k_len,  self.num_heads, self.d_k)
        q = self.query_layer(q).view(batch_size, q_len,  self.num_heads, self.d_k)
        v = self.value_layer(v).view(batch_size, v_len,  self.num_heads, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attention = scaled_dot_product_attention(q, k, self.d_k, self.mask)
        output = torch.matmul(attention, v)
        output = self.concat_projection(output.transpose(1, 2).contiguous().view(batch_size, q_len, -1))
        return self.dropout(self.layer_norm(output+residual))

class PositionWiseFFN(nn.Module): 
    def __init__(self, d_model, d_ff, dropout) -> None:
        super(PositionWiseFFN, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(d_model, d_ff, bias=True),nn.ReLU())
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        x = self.fc2(self.fc1(x))        
        return self.dropout(self.layer_norm(x+residual))
