import torch
from torch import nn, Tensor
import math

N_TOKENS = 2048
N_SPECIAL_TOKENS = 2


class BERT(nn.Module):
    def __init__(self, d_model, n_head, dim_feed, dropout, layers, max_len, output_units):
        super().__init__()

        # Define model components
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feed,
                                                         dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.d_model = d_model
        self.output_units = output_units

        self.classification = nn.Linear(d_model, output_units)
        self.masked_language_model = nn.Linear(d_model, N_TOKENS)

    def forward(self, embedding):
        #todo embedding
        #embedding = PositionalEncoding(embedding * math.sqrt(self.d_model))
        output = self.transformer(self.pos_encoder(embedding))
        y_cls = self.classification(output[:, 0, :])
        y_mlm = self.masked_language_model(output)
        return y_cls, y_mlm

class PositionalEncoding(nn.Module):
    #todo swap axis
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class BERTWithEmbedding(BERT):
    def __init__(self, d_model, n_head, dim_feed, dropout, layers, max_len, output_units):
        super().__init__(d_model, n_head, dim_feed, dropout, layers, max_len, output_units)

        self.embedding = nn.Embedding(N_SPECIAL_TOKENS + N_TOKENS, self.d_model)

    def forward(self, x):
        return super().forward(self.embedding(x))
