import pickle as pkl

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def data_load(path_text_long, path_text_short, path_num, seq_len):

    with open(path_text_long, "rb") as f:
        text_data_long = pkl.load(f)

    with open(path_text_short, "rb") as f:
        text_data_short = pkl.load(f)

    with open(path_num, "rb") as f:
        num_data = pkl.load(f)

    num_train, num_test, text_train_long, text_test_long = train_test_split(
        num_data, text_data_long, shuffle=False, test_size=None, train_size=854
    )

    (
        num_val,
        num_test,
        text_val_long,
        text_test_long,
        text_val_short,
        text_test_short,
    ) = train_test_split(
        num_test,
        text_test_long,
        text_data_short[-1],
        shuffle=False,
        test_size=None,
        train_size=241,
    )
    scaler_x_train = MinMaxScaler()
    scaler_y_train = MinMaxScaler()
    scaler_x_val = MinMaxScaler()
    scaler_y_val = MinMaxScaler()

    x_num = num_train[:, 0::4]
    x_num_val = num_val[:, 0::4]
    x_num_test = num_test[:, 0::4]

    y_num = num_train[:, 3::4]
    y_num_val = num_val[:, 3::4]
    y_num_test = num_test[:, 3::4]

    x_num_train_long = scaler_x_train.fit_transform(x_num)
    y_num_train_long = scaler_y_train.fit_transform(y_num)
    x_num_val = scaler_x_val.fit_transform(x_num_val)
    y_num_val = scaler_y_val.fit_transform(y_num_val)

    x_num_train_short = []
    y_num_train_short = []
    start = 0
    end = 0
    for text in text_data_short[:-1]:
        end = start + len(text)
        x_num_train_short.append(x_num_train_long[start:end])
        y_num_train_short.append(y_num_train_long[start:end])
        start = end - seq_len - 1

    return (
        text_train_long,
        x_num_train_long,
        y_num_train_long,
        text_data_short[:-1],
        x_num_train_short,
        y_num_train_short,
        text_val_long,
        text_val_short,
        x_num_val,
        y_num_val,
        text_test_long,
        text_test_short,
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


class Long_StockPriceEstimator(nn.Module):
    def __init__(
        self,
        textual_dim=1000,
        numerical_dim=10,
        dense_out_dim=500,
        lstm_out_dim=10,
        lstm_num_layers=1,
        drop_out=0.5,
    ):
        super(Long_StockPriceEstimator, self).__init__()
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


class Short_StockPriceEstimator(nn.Module):
    def __init__(
        self,
        textual_dim=500,
        numerical_dim=10,
        dense_out_dim=250,
        lstm_out_dim=10,
        lstm_num_layers=1,
        drop_out=0.5,
    ):
        super(Short_StockPriceEstimator, self).__init__()
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
        short_numerical_dim=10,
        numerical_dim=10,
        lstm_out_dim=10,
        lstm_num_layers=1,
        drop_out=0.5,
    ):
        super(StockPricePredictor, self).__init__()
        self.lstm = nn.LSTM(
            numerical_dim * 3,
            lstm_out_dim,
            num_layers=lstm_num_layers,
            dropout=drop_out,
            batch_first=False,
        )

    def forward(self, long_numeric_vectors, short_numeric_vectors, numeric_vectors):

        dense_outs = torch.cat(
            [long_numeric_vectors, short_numeric_vectors, numeric_vectors], dim=-1
        )
        out, hidden = self.lstm(dense_outs)

        return out[-1]
