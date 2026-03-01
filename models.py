import math
import torch
import torch.nn as nn


class QAModelV1(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_units, max_context_len):
        super().__init__()
        self.max_context_len = max_context_len
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm_context = nn.LSTM(
            embed_dim, lstm_units, bidirectional=True, batch_first=True
        )
        self.lstm_question = nn.LSTM(
            embed_dim, lstm_units, bidirectional=True, batch_first=True
        )
        self.dense_start = nn.Linear(lstm_units * 4, 1)
        self.dense_end = nn.Linear(lstm_units * 4, 1)

    def forward(self, context, question):
        context_emb = self.embedding(context)
        question_emb = self.embedding(question)
        context_enc, _ = self.lstm_context(context_emb)
        question_enc, _ = self.lstm_question(question_emb)
        question_rep = (
            question_enc[:, -1, :].unsqueeze(1).expand(-1, self.max_context_len, -1)
        )
        merged = torch.cat([context_enc, question_rep], dim=-1)
        return self.dense_start(merged).squeeze(-1), self.dense_end(merged).squeeze(-1)


class QAModelV2(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_units, max_context_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm_context = nn.LSTM(
            embed_dim, lstm_units, bidirectional=True, batch_first=True, dropout=0.3
        )
        self.lstm_question = nn.LSTM(
            embed_dim, lstm_units, bidirectional=True, batch_first=True, dropout=0.3
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_units * 2, num_heads=4, batch_first=True, dropout=0.1
        )
        self.dense1 = nn.Linear(lstm_units * 4, lstm_units * 2)
        self.dropout = nn.Dropout(0.3)
        self.dense_start = nn.Linear(lstm_units * 2, 1)
        self.dense_end = nn.Linear(lstm_units * 2, 1)
        self.relu = nn.ReLU()

    def forward(self, context, question):
        context_emb = self.embedding(context)
        question_emb = self.embedding(question)
        context_enc, _ = self.lstm_context(context_emb)
        question_enc, _ = self.lstm_question(question_emb)
        attn_output, _ = self.attention(context_enc, question_enc, question_enc)
        merged = torch.cat([context_enc, attn_output], dim=-1)
        hidden = self.dropout(self.relu(self.dense1(merged)))
        return self.dense_start(hidden).squeeze(-1), self.dense_end(hidden).squeeze(-1)


class QAModelV3(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.ctx_encoder = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
        )
        self.q_encoder = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 8, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_size * 2),
        )
        self.span_modeling = nn.LSTM(
            hidden_size * 2, hidden_size, bidirectional=True, batch_first=True
        )
        self.start_head = nn.Linear(hidden_size * 2, 1)
        self.end_head = nn.Linear(hidden_size * 2, 1)
        self.no_answer_head = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, 1),
        )

    def forward(self, context_ids, question_ids):
        c = self.embedding(context_ids)
        q = self.embedding(question_ids)
        c_enc, _ = self.ctx_encoder(c)
        q_enc, _ = self.q_encoder(q)

        sim = torch.bmm(c_enc, q_enc.transpose(1, 2)) / math.sqrt(c_enc.size(-1))
        c2q_attn = torch.softmax(sim, dim=-1)
        c2q = torch.bmm(c2q_attn, q_enc)

        q2c_scores = torch.softmax(torch.max(sim, dim=-1).values, dim=-1)
        q2c_vec = torch.bmm(q2c_scores.unsqueeze(1), c_enc)
        q2c = q2c_vec.expand(-1, c_enc.size(1), -1)

        merged = torch.cat([c_enc, c2q, c_enc * c2q, c_enc * q2c], dim=-1)
        fused = self.fusion(merged)
        modeled, _ = self.span_modeling(fused)

        start_logits = self.start_head(modeled).squeeze(-1)
        end_logits = self.end_head(modeled).squeeze(-1)

        cls_ctx = modeled[:, 0, :]
        q_summary = q_enc[:, -1, :]
        no_answer_logit = self.no_answer_head(
            torch.cat([cls_ctx, q_summary], dim=-1)
        ).squeeze(-1)
        return start_logits, end_logits, no_answer_logit


class QAModelV5(nn.Module):
    """
    V5: version plus profonde que V3 (toujours from-scratch).
    - encoders 2 couches biLSTM
    - fusion attention context-question
    - 2 couches de modeling pour les spans
    - tête no-answer séparée
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(0.2)

        self.ctx_encoder = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
        )
        self.q_encoder = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 8, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.LayerNorm(hidden_size * 2),
        )

        self.span_modeling_1 = nn.LSTM(
            hidden_size * 2,
            hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.span_modeling_2 = nn.LSTM(
            hidden_size * 2,
            hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.model_dropout = nn.Dropout(0.2)

        self.start_head = nn.Linear(hidden_size * 2, 1)
        self.end_head = nn.Linear(hidden_size * 2, 1)

        self.no_answer_head = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, 1),
        )

    def forward(self, context_ids, question_ids):
        c = self.emb_dropout(self.embedding(context_ids))
        q = self.emb_dropout(self.embedding(question_ids))

        c_enc, _ = self.ctx_encoder(c)
        q_enc, _ = self.q_encoder(q)

        sim = torch.bmm(c_enc, q_enc.transpose(1, 2)) / math.sqrt(c_enc.size(-1))
        c2q_attn = torch.softmax(sim, dim=-1)
        c2q = torch.bmm(c2q_attn, q_enc)

        q2c_scores = torch.softmax(torch.max(sim, dim=-1).values, dim=-1)
        q2c_vec = torch.bmm(q2c_scores.unsqueeze(1), c_enc)
        q2c = q2c_vec.expand(-1, c_enc.size(1), -1)

        merged = torch.cat([c_enc, c2q, c_enc * c2q, c_enc * q2c], dim=-1)
        fused = self.fusion(merged)

        m1, _ = self.span_modeling_1(fused)
        m2, _ = self.span_modeling_2(m1)
        modeled = self.model_dropout(m2)

        start_logits = self.start_head(modeled).squeeze(-1)
        end_logits = self.end_head(modeled).squeeze(-1)

        cls_ctx = modeled[:, 0, :]
        q_summary = q_enc[:, -1, :]
        no_answer_logit = self.no_answer_head(
            torch.cat([cls_ctx, q_summary], dim=-1)
        ).squeeze(-1)
        return start_logits, end_logits, no_answer_logit
