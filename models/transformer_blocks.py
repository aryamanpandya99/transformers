from modules import MultiHeadAttention, PositionWiseFFN, positional_embedding
from torch import nn

class TransformerLayer(nn.Module): 
    def __init__(self, d_k, d_model, d_v, num_heads, d_ff, mask, dropout) -> None:
        super(TransformerLayer, self).__init__()
        self.k_layer, self.q_layer, self.v_layer = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.multihead_attention = MultiHeadAttention(d_k, d_model, d_v, dropout, num_heads, mask)
        self.pointwise_ffn = PositionWiseFFN(d_model, d_ff, dropout)

    def forward(self, x): 
        k, q, v = self.k_layer(x), self.q_layer(x), self.v_layer(x)
        output = self.multihead_attention(q, k, v)
        return self.pointwise_ffn(output)


class Transformer(nn.Module):
    """
    If mask = False, this is an Encoder and if mask = True
    this is a decoder for a decoder only transformer
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
        x = self.dropout(embedded + self.positional_embedding(x, self.d_model))
        for layer in self.layers:
            x = layer(x)
        return x

class DecoderLayer(nn.Module):
    """
    This module is designed as per the Attention is All You Need paper,
    assuming that we're propagating some output from the encoder to the decoder.
    For decoder only models, we can get rid of the cross attention module
    and the only difference between the encoder and decoder in this case
    would be the masking of future tokens
    """
    def __init__(self, d_k, d_model, d_v, num_heads, d_ff, dropout) -> None:
        super(DecoderLayer, self).__init__()
        self.k_layer, self.q_layer, self.v_layer = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.k_layer2, self.q_layer2, self.v_layer2 = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.masked_multihead_attention = MultiHeadAttention(d_k, d_model, d_v, dropout, num_heads, mask=True)
        self.multihead_attention = MultiHeadAttention(d_k, d_model, d_v, dropout, num_heads, mask=False)
        self.positiontwise_ffn = PositionWiseFFN(d_model, d_ff, dropout)

    def forward(self, outputs, encoder_output):
        k1, q1, v1 = self.k_layer(outputs), self.q_layer(outputs), self.v_layer(outputs)
        outputs = self.masked_multihead_attention(q1, k1, v1, True)
        k2, q2, v2 = self.k_layer2(encoder_output), self.q_layer2(outputs), self.v_layer2(encoder_output)  # query comes from outputs of previous decoder layer while k, v come from the encoder's output
        outputs = self.multihead_attention(q2, k2, v2)
        return self.positiontwise_ffn(outputs)
    
class Decoder(nn.Module):
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