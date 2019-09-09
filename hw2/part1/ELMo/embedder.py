import numpy as np
import torch.nn as nn
import torch.utils.data.dataloader
import torch

class Embedder:
    
    def __init__(self, n_ctx_embs, ctx_emb_dim):
        dictionary_size = 120000
        n_ctx_embs = 128
        ctx_emb_dim = 256

        # forward net
        self.lstm_forward_1 = torch.nn.LSTM(input_size=n_ctx_embs, hidden_size=ctx_emb_dim)
        self.linear_forward_1 = torch.nn.Linear(ctx_emb_dim, int(ctx_emb_dim / 2))

        self.lstm_forward_2 = torch.nn.LSTM(input_size=int(ctx_emb_dim / 2), hidden_size=ctx_emb_dim)
        self.linear_forward_2 = torch.nn.Linear(ctx_emb_dim, dictionary_size)

        # backward net
        self.lstm_backward_1 = torch.nn.LSTM(input_size=n_ctx_embs, hidden_size=ctx_emb_dim)
        self.linear_backward_1 = torch.nn.Linear(ctx_emb_dim, int(ctx_emb_dim / 2))

        self.lstm_backward_2 = torch.nn.LSTM(input_size=int(ctx_emb_dim / 2), hidden_size=ctx_emb_dim)
        self.linear_backward_2 = torch.nn.Linear(ctx_emb_dim, dictionary_size)
        self.softmax = torch.nn.Softmax()
        self.n_ctx_embs = 0
        self.ctx_emb_dim = 0

    def __call__(self, sentences, max_sent_len):
        forward, (_, _) = self.lstm_forward_1(sentences)
        forward = self.linear_forward_1(forward)
        forward, (_, _) = self.lstm_forward_2(forward)
        forward = self.linear_forward_2(forward)
        forward = self.softmax(forward)

        backward, (_, _) = self.lstm_forward_1(sentences)
        backward = self.linear_forward_1(backward)
        backward, (_, _) = self.lstm_forward_2(backward)
        backward = self.linear_forward_2(backward)
        backward = self.softmax(backward)

        return np.zeros(
            (len(sentences), min(max(map(len, sentences)), max_sent_len), self.n_ctx_embs, self.ctx_emb_dim), dtype=np.float32)
