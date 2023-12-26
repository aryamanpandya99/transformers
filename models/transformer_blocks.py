"""
Implementations of different versions of transformer layers/blocks
Aryaman Pandya
"""

from modules import MultiHeadAttention, PositionWiseFFN, positional_embedding
from torch import nn

class TransformerLayer(nn.Module):
    """
    Class implementation of a generic single self-attention only transformer layer
    """
    def __init__(self, d_k, d_model, d_v, num_heads, d_ff, mask, dropout) -> None:
        super(TransformerLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_k=d_k, d_model=d_model, d_v=d_v, dropout=dropout, num_heads=num_heads, mask=mask)
        self.pointwise_ffn = PositionWiseFFN(d_model, d_ff, dropout)

    def forward(self, x):
        output = self.multihead_attention(x)
        return self.pointwise_ffn(output)


class Transformer(nn.Module):
    """
    If mask = False, this is an Encoder and if mask = True
    this is a decoder for a decoder only transformer (no cross attention)
    """
    def __init__(self, d_k, d_model, d_v, d_ff, num_heads, num_layers, vocab_size, mask=False, dropout=0.1) -> None:
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_embedding = positional_embedding
        self.dropout = nn.Dropout(dropout)
        self.layers = [TransformerLayer(d_k, d_model, d_v, num_heads, d_ff, mask, dropout) for _ in range(num_layers)]

    def forward(self, x):
        embedded = self.embedding(x)
        x = self.dropout(embedded + self.positional_embedding(x, self.d_model).to(x.device))
        for layer in self.layers:
            layer = layer.to(x.device)
            x = layer(x)
        return x

class DecoderLayer(nn.Module):
    """
    Implementation of a decoder layer containing a cross attention layer
    """
    def __init__(self, d_k, d_model, d_v, num_heads, d_ff, dropout) -> None:
        super(DecoderLayer, self).__init__()
        self.masked_multihead_attention = MultiHeadAttention(d_k, d_model, d_v, dropout, num_heads, mask=True)
        self.multihead_attention = MultiHeadAttention(d_k, d_model, d_v, dropout, num_heads, mask=False)
        self.positiontwise_ffn = PositionWiseFFN(d_model, d_ff, dropout)

    def forward(self, outputs, encoder_output):
        outputs = self.masked_multihead_attention(outputs)  # self attention
        outputs = self.multihead_attention(outputs, encoder_output)  # cross attention
        return self.positiontwise_ffn(outputs)
    
class Decoder(nn.Module):
    """
    Implementation of a decoder assuming cross attention
    """
    def __init__(self, d_k, d_model, d_v, d_ff, num_heads, num_layers, vocab_size, out_dims, dropout=0.1) -> None:
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_embedding = positional_embedding
        self.dropout = nn.Dropout(dropout)
        self.layers = [DecoderLayer(d_k, d_model, d_v, num_heads, d_ff, dropout) for _ in range(num_layers)]
        self.lm_head = nn.Linear(d_model, out_dims)

    def forward(self, x, encoder_output):
        embedded = self.embedding(x)
        x = self.dropout(embedded + self.positional_embedding(x, self.d_model).to(x.device))
        for layer in self.layers:
            layer = layer.to(x.device)
            x = layer(x, encoder_output)
        return self.lm_head(self.dropout(x))