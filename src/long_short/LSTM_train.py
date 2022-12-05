import argparse
import datetime
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import tqdm
from dateutil.relativedelta import relativedelta
from LSTM import (
    Long_StockPriceEstimator,
    Short_StockPriceEstimator,
    StockPricePredictor,
    data_load,
    prepare_data,
)
from monthdelta import monthmod

start = datetime.date(2015, 1, 1)
end = datetime.date(2018, 7, 1)

path_long_pkl = "./../../data/pkl/long/"
path_long_model = "./../../data/model/long/"
path_short_pkl = "./../../data/pkl/short/"
path_short_model = "./../../data/model/short/"
path_longshort_model = "./../../data/model/long_short/"
path_data = "./../../data/"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num", default="car_num.pkl")
    parser.add_argument("-t", "--text", default="car_text.pkl")

    parser.add_argument("-e", "--epoch", default=1000, type=int)
    parser.add_argument("-b", "--batch_size", default=30, type=int)
    parser.add_argument("-s", "--seq_len", default=20, type=int)

    parser.add_argument("-stock", "--stock", default="TOPIX10C_CAR")

    parser.add_argument("-e_model", "--estimated_model", default="estimated_car.model")

    parser.add_argument(
        "-p_model",
        "--predicted_model",
        default="predicted_car.model",
    )

    parser.add_argument("-m", "--month", default=12)

    return parser.parse_args()


def read_csv(path_stock):

    file_names = path_stock.glob("*.csv")

    date = []
    for f in file_names:
        stock_df = pd.read_csv(f, encoding="CP932")
        if len(stock_df) == 1219:
            date.extend([int(t.replace("-", "")) for t in stock_df["日付"]])
    date_list = sorted(list(set(date)))

    return date_list


def long_estimate(
    X_train_n, X_train_t, y_train, seq_len, batch_size, epochs, path_long_model
):

    estimated_lstm = Long_StockPriceEstimator(
        textual_dim=len(X_train_t[0]),
        numerical_dim=len(X_train_n[0]),
        dense_out_dim=int(len(X_train_t[0]) / 2),
        lstm_out_dim=len(X_train_n[0]),
    ).to(device)

    estimated_lstm.load_state_dict(torch.load(path_long_model))

    estimation = []

    idx = np.arange(seq_len, X_train_n.shape[0] + 1)

    for t_i in range(0, len(idx), batch_size):
        batch_idx = idx[t_i : (t_i + batch_size)]
        feats_n = prepare_data(
            seq_len,
            batch_idx,
            torch.tensor(
                X_train_n.reshape(-1, len(X_train_n[0])),
                dtype=torch.float,
                device=device,
            ),
            device,
        )

        feats_t = prepare_data(
            seq_len,
            batch_idx,
            torch.tensor(X_train_t, dtype=torch.float, device=device),
            device,
        )
        _, out = estimated_lstm.forward(feats_t, feats_n)
        estimation.append(out)

    return estimation


def short_estimate(
    X_train_n, X_train_t, y_train, seq_len, batch_size, epochs, path_short_model
):

    estimated_lstm = Short_StockPriceEstimator(
        textual_dim=len(X_train_t[0]),
        numerical_dim=len(X_train_n[0]),
        dense_out_dim=int(len(X_train_t[0]) / 2),
        lstm_out_dim=len(X_train_n[0]),
    ).to(device)

    estimated_lstm.load_state_dict(torch.load(path_short_model))
    estimation = []

    idx = np.arange(seq_len, X_train_n.shape[0] + 1)

    for t_i in range(0, len(idx), batch_size):
        batch_idx = idx[t_i : (t_i + batch_size)]
        feats_n = prepare_data(
            seq_len,
            batch_idx,
            torch.tensor(
                X_train_n.reshape(-1, len(X_train_n[0])),
                dtype=torch.float,
                device=device,
            ),
            device,
        )

        feats_t = prepare_data(
            seq_len,
            batch_idx,
            torch.tensor(X_train_t, dtype=torch.float, device=device),
            device,
        )
        _, out = estimated_lstm.forward(feats_t, feats_n)
        estimation.append(out)

    return estimation


def train(args):

    min_loss = 999
    seq_len = args.seq_len
    epochs = args.epoch
    batch_size = args.batch_size

    path_stock = pathlib.Path(path_data + args.stock)
    date_list = read_csv(path_stock)
    month_datetime = relativedelta(months=int(args.month))
    month = int(args.month)
    short_estimation = []

    (
        X_train_t_long,
        X_train_n_long,
        y_train_long,
        X_train_t_short,
        X_train_n_short,
        y_train_short,
        X_val_t_long,
        X_val_t_short,
        X_val_n,
        y_val,
        _,
        _,
        _,
        _,
    ) = data_load(
        path_long_pkl + args.text,
        path_short_pkl + args.text,
        path_long_pkl + args.num,
        seq_len,
    )

    long_estimated_lstm = Long_StockPriceEstimator(
        textual_dim=len(X_train_t_long[0]),
        numerical_dim=len(X_train_n_long[0]),
        dense_out_dim=int(len(X_train_t_long[0]) / 2),
        lstm_out_dim=len(X_train_n_long[0]),
    ).to(device)

    short_estimated_lstm = Short_StockPriceEstimator(
        textual_dim=len(X_train_t_short[0][0]),
        numerical_dim=len(X_train_n_short[0][0]),
        dense_out_dim=int(len(X_train_t_short[0][0]) / 2),
        lstm_out_dim=len(X_train_n_short[0][0]),
    ).to(device)

    predicated_lstm = StockPricePredictor(
        short_numerical_dim=len(X_train_n_long[0]),
        numerical_dim=len(X_train_n_long[0]),
        lstm_out_dim=len(X_train_n_long[0]),
    ).to(device)

    optimizer = optim.Adam(predicated_lstm.parameters())

    date_start = start
    date_to = date_start + month_datetime
    date_list = pd.to_datetime(date_list, format="%Y%m%d")

    for i in range(int(monthmod(start, end)[0].months / month)):

        date_end = list(filter(lambda x: x > date_to, date_list.date))[seq_len - 2]

        if monthmod(date_to, end)[0].months < month:
            date_end = end - relativedelta(days=1)

        date_list_term = date_list[
            (str(date_start) <= date_list) & (date_list <= str(date_end))
        ].date
        date_list_term = [int(str(date).replace("-", "")) for date in date_list_term]

        term = str(date_start) + "_" + str(date_end)

        path_model = path_short_model + args.estimated_model
        path_model = (
            path_model[: path_model.find(".model")]
            + "_"
            + str(term)
            + path_model[path_model.find(".model") :]
        )

        out = short_estimate(
            X_train_n_short[i],
            X_train_t_short[i],
            y_train_short[i],
            seq_len,
            batch_size,
            epochs,
            path_model,
        )
        short_estimation.extend(out)

    short_estimation = torch.cat(short_estimation, dim=1)
    short_estimation = torch.split(short_estimation, batch_size, dim=1)
    long_estimation = long_estimate(
        X_train_n_long,
        X_train_t_long,
        y_train_long,
        seq_len,
        batch_size,
        epochs,
        path_long_model + args.estimated_model,
    )

    long_estimated_lstm.load_state_dict(
        torch.load(path_long_model + args.estimated_model)
    )
    short_estimated_lstm.load_state_dict(torch.load(path_model))

    for _ in tqdm.tqdm(range(epochs)):
        idx = np.arange(seq_len, X_train_n_long.shape[0] + 1)
        tmp = 0
        i = 0
        for t_i in range(0, len(idx), batch_size):
            batch_idx = idx[t_i : (t_i + batch_size)]

            feats_n = prepare_data(
                seq_len,
                batch_idx,
                torch.tensor(
                    X_train_n_long.reshape(-1, len(X_train_n_long[0])),
                    dtype=torch.float,
                    device=device,
                ),
                device,
            )

            y_target = torch.tensor(
                y_train_long[batch_idx - 1], dtype=torch.float, device=device
            )

            optimizer.zero_grad()
            y_hat = predicated_lstm.forward(
                long_estimation[i], short_estimation[i], feats_n
            )
            loss = (y_target - y_hat).pow(2).mean()
            tmp += loss
            loss.backward(retain_graph=True)
            optimizer.step()
            i += 1

        feats_n = prepare_data(
            seq_len,
            np.arange(seq_len, X_val_n.shape[0] + 1),
            torch.tensor(
                X_val_n.reshape(-1, len(X_val_n[0])), dtype=torch.float, device=device
            ),
            device,
        )

        feats_t_long = prepare_data(
            seq_len,
            np.arange(seq_len, X_val_n.shape[0] + 1),
            torch.tensor(X_val_t_long, dtype=torch.float, device=device),
            device,
        )
        feats_t_short = prepare_data(
            seq_len,
            np.arange(seq_len, X_val_n.shape[0] + 1),
            torch.tensor(X_val_t_short, dtype=torch.float, device=device),
            device,
        )

        _, out_long = long_estimated_lstm.forward(feats_t_long, feats_n)
        _, out_short = short_estimated_lstm.forward(feats_t_short, feats_n)

        y_hat = predicated_lstm.forward(out_long, out_short, feats_n)
        pre = y_hat.view(-1).to("cpu").detach().numpy().reshape(-1, len(X_val_n[0]))
        tmp = 0
        tmp = np.mean((y_val[(seq_len - 1) :] - pre) ** 2)
        if tmp < min_loss:
            min_loss = tmp
            torch.save(
                predicated_lstm.state_dict(),
                path_longshort_model + args.predicted_model,
            )


if __name__ == "__main__":
    args = parser_args()
    train(args)
