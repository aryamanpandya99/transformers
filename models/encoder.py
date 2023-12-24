from modules import MultiHeadAttention, PositionWiseFFN, positional_embedding
from torch import nn

class EncoderLayer(nn.Module): 
    def __init__(self, d_k, d_model, d_v, num_heads, d_ff, dropout) -> None:
        super(EncoderLayer, self).__init__()
        self.k_layer, self.q_layer, self.v_layer = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.multihead_attention = MultiHeadAttention(d_k, d_model, d_v, dropout, num_heads)
        self.pointwise_ffn = PositionWiseFFN(d_model, d_ff, dropout)

    def forward(self, x): 
        k, q, v = self.k_layer(x), self.q_layer(x), self.v_layer(x)
        output = self.multihead_attention(q, k, v)
        return self.pointwise_ffn(output)


class Encoder(nn.Module):
    def __init__(self, d_k, d_model, d_v, d_ff, num_heads, num_layers, vocab_size, dropout=0.1) -> None:
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_embedding = positional_embedding
        self.dropout = nn.Dropout(dropout)
        self.layers = [EncoderLayer(d_k, d_model, d_v, num_heads, d_ff, dropout) for _ in range(num_layers)]

    def forward(self, x):
        embedded = self.embedding(x)
        x = self.dropout(embedded + self.positional_embedding(x, self.d_model))
        for layer in self.layers:
            x = layer(x)
        return x
