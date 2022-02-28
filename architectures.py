import torch
from torch import nn, Tensor
import math

N_TOKENS = 2048
N_SPECIAL_TOKENS = 2
MASK_TOKEN = N_TOKENS
CLS_TOKEN = N_TOKENS + 1



class BERT(nn.Module):
    """
    Abstract superclass
    """
    def __init__(self, d_model, n_head, dim_feed, dropout, layers, max_len, output_units):
        super().__init__()

        # Define model components
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feed,
                                                         dropout=dropout, batch_first=True, layer_norm_eps=6.1e-5)
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.d_model = d_model
        self.output_units = output_units

        self.classification = nn.Linear(d_model, output_units)

    def forward(self, embedding):
        embedding = self.pos_encoder(embedding * math.sqrt(self.d_model))
        output = self.transformer(embedding)
        y_cls = self.classification(output[:, 0, :])
        y_mask_prediction = self.model_for_pretraining(output)
        return y_cls, y_mask_prediction

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)

class BERTWithEmbedding(BERT):
    """
    BERT with token input
    """
    def __init__(self, d_model, n_head, dim_feed, dropout, layers, max_len, output_units):
        super().__init__(d_model, n_head, dim_feed, dropout, layers, max_len, output_units)

        self.embedding = nn.Embedding(N_SPECIAL_TOKENS + N_TOKENS, self.d_model)
        self.model_for_pretraining = nn.Linear(d_model, N_TOKENS)

    def forward(self, x):
        return super().forward(self.embedding(x))

class BERTWithoutEmbedding(BERT):
    """
    With linear projection (described by MusiCoder in section 3.1)
    """
    def __init__(self, d_model, n_head, dim_feed, dropout, layers, max_len, output_units, input_units):
        super().__init__(d_model, n_head, dim_feed, dropout, layers, max_len, output_units)

        self.input_units = input_units
        self.norm = torch.nn.LayerNorm(input_units, eps=6.1e-4, elementwise_affine=False)
        self.projection = nn.Linear(input_units, d_model)
        self.model_for_pretraining = nn.Linear(d_model, input_units)

    def forward(self, x):
        x = self.norm(x)
        return super().forward(self.projection(x))

class BERTWithCodebooks(BERTWithoutEmbedding):
    def __init__(self, d_model, n_head, dim_feed, dropout, layers, max_len, output_units, input_units):
        super().__init__(d_model, n_head, dim_feed, dropout, layers, max_len, output_units, input_units)

        # Replace model for pretraining so that token classification is possible
        self.model_for_pretraining = nn.Linear(d_model, N_TOKENS)

        codebooks = torch.load("./codebooks.pt")
        # Append zeros for special tokens
        zeros = torch.zeros((2, 64))
        codebooks = torch.cat((codebooks, zeros), 0)
        self.register_buffer("codebooks", codebooks)

    def forward(self, x):
        # Replace with codebooks
        x = self.codebooks[x.long(), :]
        return super().forward(x)
