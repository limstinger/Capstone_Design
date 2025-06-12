import torch
import torch.nn as nn
import torch.nn.functional as F
from performer_pytorch import Performer

# ① TransformerLayerWithAttn: 어텐션 가중치 반환
class TransformerLayerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model,
                                               num_heads=nhead,
                                               dropout=dropout,
                                               batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # src: (batch, seq_len, d_model)
        attn_output, attn_weights = self.self_attn(src, src, src, need_weights=True)
        # attn_weights: (batch, nhead, seq_len, seq_len)

        src2 = self.norm1(src + self.dropout1(attn_output))
        ff = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src3 = self.norm2(src2 + self.dropout2(ff))
        return src3, attn_weights


# ② FusionLSTMTransformerWithAttn: LSTM → Transformer(어텐션) → Classifier
class FusionLSTMTransformerWithAttn(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, nhead):
        super().__init__()
        self.lstm = nn.LSTM(input_dim,
                            hidden_dim,
                            num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        d_model = hidden_dim * 2
        self.transformer_layer = TransformerLayerWithAttn(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)   # 이진 분류 기준
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        h, _ = self.lstm(x)                     # (batch, seq_len, hidden_dim*2)
        out_t, attn_weights = self.transformer_layer(h)
        # out_t: (batch, seq_len, hidden_dim*2)
        # attn_weights: (batch, nhead, seq_len, seq_len)

        pooled = out_t.mean(dim=1)              # (batch, hidden_dim*2)
        logits = self.classifier(pooled)        # (batch, 2)
        return logits, attn_weights
    

class FusionLSTMPerformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, heads, depth, max_seq_len):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        d_model = hidden_dim * 2

        self.performer = Performer(
            dim=d_model,
            dim_head=d_model // heads,  # ← 추가 (헤드당 차원 수)
            depth=depth,
            heads=heads,
            causal=False,
            ff_dropout=dropout,
            attn_dropout=dropout,
            nb_features=256
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        h, _ = self.lstm(x)         # (batch, seq_len, hidden_dim*2)
        out = self.performer(h)     # (batch, seq_len, hidden_dim*2)
        pooled = out.mean(dim=1)    # mean pooling
        logits = self.classifier(pooled)
        return logits, None