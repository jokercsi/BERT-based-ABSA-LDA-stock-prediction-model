import pickle as pkl

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


def data_load(path_text, path_num):

    with open(path_text, "rb") as f:
        text_data = pkl.load(f)

    with open(path_num, "rb") as f:
        num_data = pkl.load(f)

    num_train, num_test, text_train, text_test = train_test_split(
        num_data, text_data, shuffle=False, test_size=None, train_size=454
    )

    num_val, num_test, text_val, text_test = train_test_split(
        num_test, text_test, shuffle=False, test_size=None, train_size=194
    )
    x_num_train = num_train[:, 0::4]
    x_num_val = num_val[:, 0::4]
    x_num_test = num_test[:, 0::4]

    y_num_train = num_train[:, 3::4]
    y_num_val = num_val[:, 3::4]
    y_num_test = num_test[:, 3::4]

    return (
        text_train,
        x_num_train,
        y_num_train,
        text_val,
        x_num_val,
        y_num_val,
        text_test,
        x_num_test,
        y_num_test,
    )


def prepare_data(seq_len, batch_idx, X_data, device):
    feats = torch.zeros(
        (seq_len, len(batch_idx), X_data.shape[1]), dtype=torch.float, device=device
    )
    for b_i, b_idx in enumerate(batch_idx):
        b_slc = slice(b_idx - seq_len, b_idx)
        feats[:, b_i, :] = X_data[b_slc, :]
    return feats


class StockPriceEstimator(nn.Module):
    def __init__(
        self,
        textual_dim=1000,
        numerical_dim=10,
        dense_out_dim=500,
        lstm_out_dim=10,
        lstm_num_layers=1,
        drop_out=0.5,
    ):
        super(StockPriceEstimator, self).__init__()
        self.dense_text = nn.Linear(textual_dim, dense_out_dim)
        self.dense_numeric = nn.Linear(numerical_dim, dense_out_dim)

        self.lstm = nn.LSTM(
            dense_out_dim * 2,
            lstm_out_dim,
            num_layers=lstm_num_layers,
            dropout=drop_out,
            batch_first=False,
        )

    def forward(self, text_vectors, numeric_vectors):
        text_dense_outs = []
        numeric_dense_outs = []
        for i in range(text_vectors.size(0)):
            temp_out = self.dense_text(text_vectors[i])
            text_dense_outs.append(temp_out)

        for i in range(numeric_vectors.size(0)):
            temp_out = self.dense_numeric(numeric_vectors[i])
            numeric_dense_outs.append(temp_out)

        text_dense_outs = torch.stack(text_dense_outs, dim=0)
        numeric_dense_outs = torch.stack(numeric_dense_outs, dim=0)

        dense_outs = torch.cat([text_dense_outs, numeric_dense_outs], dim=-1)
        out, hidden = self.lstm(dense_outs)

        return out[-1], out


class StockPricePredictor(nn.Module):
    def __init__(
        self,
        numerical_dim=10,
        lstm_out_dim=10,
        lstm_num_layers=1,
        drop_out=0.5,
    ):
        super(StockPricePredictor, self).__init__()
        self.lstm = nn.LSTM(
            numerical_dim + numerical_dim,
            lstm_out_dim,
            num_layers=lstm_num_layers,
            dropout=drop_out,
            batch_first=False,
        )

    def forward(self, estimated_numeric_vectors, numeric_vectors):

        dense_outs = torch.cat([estimated_numeric_vectors, numeric_vectors], dim=-1)
        out, hidden = self.lstm(dense_outs)

        return out[-1]
