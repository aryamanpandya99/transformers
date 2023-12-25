from modules import MultiHeadAttention, PositionWiseFFN, positional_embedding
from torch import nn

class DecoderLayer(nn.Module):
    def __init__(self, d_k, d_model, d_v, num_heads, d_ff, dropout) -> None:
        super(DecoderLayer, self).__init__()
        self.k_layer, self.q_layer, self.v_layer = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.k_layer2, self.q_layer2, self.v_layer2 = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.masked_multihead_attention = MultiHeadAttention(d_k, d_model, d_v, dropout, num_heads)
        self.multihead_attention = MultiHeadAttention(d_k, d_model, d_v, dropout, num_heads)
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
        self.fc = nn.Linear(d_model, out_dims)

    def forward(self, x, encoder_output):
        embedded = self.embedding(x)
        x = self.dropout(embedded + self.positional_embedding(x, self.d_model).to(x.device))
        for layer in self.layers:
            layer = layer.to(x.device)
            x = layer(x, encoder_output)
        return self.fc(self.dropout(x))