import argparse
import os
import pathlib
import pickle as pkl

import numpy as np
import torch
from LSTM import (
    Long_StockPriceEstimator,
    Short_StockPriceEstimator,
    StockPricePredictor,
    data_load,
    prepare_data,
)
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

path_long_pkl = "./../../data/pkl/long/"
path_long_model = "./../../data/model/long/"
path_short_pkl = "./../../data/pkl/short/"
path_short_model = "./../../data/model/short/"
path_longshort_pkl = "./../../data/pkl/long_short/"
path_longshort_model = "./../../data/model/long_short/"
path_data = "./../../data/"


def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num", default="car_num.pkl")
    parser.add_argument("-t", "--text", default="car_text.pkl")

    parser.add_argument("-stock", "--stock", default="TOPIX10C_CAR")
    parser.add_argument("-graph", "--graph", default="./../../data/result/long_short")

    parser.add_argument("-s", "--seq_len", default=20, type=int)

    parser.add_argument(
        "-l_emodel", "--long_estimated_model", default="estimated_car.model"
    )

    parser.add_argument(
        "-s_emodel",
        "--short_estimated_model",
        default="estimated_car_2017-01-01_2018-06-30.model",
    )
    parser.add_argument(
        "-p_model",
        "--predicted_model",
        default="predicted_car.model",
    )

    parser.add_argument("-o", "--output", default="predict_car.pkl")

    return parser.parse_args()


def read_company_id(path_stock):

    id_name = os.listdir(path_stock)
    name = [
        f.replace(".csv", "")
        for f in id_name
        if os.path.isfile(os.path.join(path_stock, f))
    ]

    return name


def test_data_load(path_num):

    with open(path_num, "rb") as f:
        num_data = pkl.load(f)

    num_train, num_test = train_test_split(
        num_data, shuffle=False, test_size=None, train_size=854
    )
    num_val, num_test = train_test_split(
        num_test, shuffle=False, test_size=None, train_size=241
    )

    open_price = num_test[:, 0::4]
    low_price = num_test[:, 1::4]
    high_price = num_test[:, 2::4]
    close_price = num_test[:, 3::4]

    return open_price, low_price, high_price, close_price


def mcfd(pre, close_price):

    pre_future = pre[1:, :]
    pre_curr = pre[:-1, :]
    pred = np.log(pre_future) - np.log(pre_curr)

    la_future = close_price[1:, :]
    la_curr = close_price[:-1, :]
    lab = np.log(la_future) - np.log(la_curr)

    tmp = np.zeros_like(pred)
    mcfd = np.zeros((1, tmp.shape[1]))

    for i in range(tmp.shape[1]):
        for j in range(tmp.shape[0]):
            if pred[j, i] * lab[j, i] >= 0.0:
                tmp[j, i] = 1
        mcfd[0, i] = np.mean(tmp[:, i])

    return mcfd[0]


def mftr(pre, close_price):

    pre_future = pre[1:, :]
    pre_curr = pre[:-1, :]
    pred = np.log(pre_future) - np.log(pre_curr)

    la_future = close_price[1:, :]
    la_curr = close_price[:-1, :]
    lab = np.log(la_future) - np.log(la_curr)

    tmp = np.zeros_like(pred)
    mftr = np.zeros((1, tmp.shape[1]))
    for i in range(tmp.shape[1]):
        for j in range(tmp.shape[0]):
            tmp[j, i] = np.sign(pred[j, i]) * lab[j, i]
        mftr[0, i] = np.mean(tmp[:, i])
    return mftr[0]


def trade(pre, open_price, low_price, high_price, close_price):

    r = (pre - open_price) / open_price
    returns = np.zeros(r.shape)

    for i in range(r.shape[1]):
        for j in range(r.shape[0]):
            if r[j, i] >= 0:
                if high_price[j, i] >= open_price[j, i] * 1.02:
                    returns[j, i] += (
                        (open_price[j, i] * 0.02) * (1000000 / open_price[j, i])
                    ) - 535
                else:
                    returns[j, i] += (
                        (-open_price[j, i] + close_price[j, i])
                        * (1000000 / open_price[j, i])
                    ) - 535

            elif r[j, i] < 0:
                if low_price[j, i] <= open_price[j, i] * 0.98:
                    returns[j, i] += (
                        (open_price[j, i] * 0.02) * (1000000 / open_price[j, i])
                    ) - 385
                else:
                    returns[j, i] += (
                        (open_price[j, i] - close_price[j, i])
                        * (1000000 / open_price[j, i])
                    ) - 385

    return sum(returns)


def result(predict_price, seq_len, path_num, path_stock, path_graph):
    company_id = read_company_id(path_stock)
    path_graph = pathlib.Path(path_graph)

    open_price, low_price, high_price, close_price = test_data_load(path_num)

    open_price = open_price[(seq_len - 1) :]
    low_price = low_price[(seq_len - 1) :]
    high_price = high_price[(seq_len - 1) :]
    close_price = close_price[(seq_len - 1) :]

    mcfd_result = mcfd(predict_price, close_price)
    mftr_result = mftr(predict_price, close_price)
    trade_result = trade(predict_price, open_price, low_price, high_price, close_price)

    for i, (company, mcfd_, mftr_, trade_) in enumerate(
        zip(company_id, mcfd_result, mftr_result, trade_result)
    ):
        print("--------------" + company + "-----------------")
        print("mcfd: ", mcfd_, "mftr:", mftr_, " returns: ", trade_)

        plt.figure(figsize=(15, 8), dpi=300)
        plt.plot(predict_price[:, i], label="predict_price")
        plt.plot(open_price[:, i], label="open_price")
        plt.plot(close_price[:, i], label="close_price")
        plt.legend(["predict_price", "open_price", "close_price"], fontsize=20)
        file_name = company + ".png"
        plt.savefig(path_graph / file_name, dpi=300)


def test(args):

    seq_len = args.seq_len

    scaler_x_test = MinMaxScaler()
    scaler_y_test = MinMaxScaler()

    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        X_test_t_long,
        X_test_t_short,
        X_test_n,
        y_test,
    ) = data_load(
        path_long_pkl + args.text,
        path_short_pkl + args.text,
        path_long_pkl + args.num,
        seq_len,
    )

    X_test_n = scaler_x_test.fit_transform(X_test_n)
    y_test = scaler_y_test.fit_transform(y_test)

    long_estimated_lstm = Long_StockPriceEstimator(
        textual_dim=len(X_test_t_long[0]),
        numerical_dim=len(X_test_n[0]),
        dense_out_dim=int(len(X_test_t_long[0]) / 2),
        lstm_out_dim=len(X_test_n[0]),
    ).to(device)

    short_estimated_lstm = Short_StockPriceEstimator(
        textual_dim=len(X_test_t_short[0]),
        numerical_dim=len(X_test_n[0]),
        dense_out_dim=int(len(X_test_t_short[0]) / 2),
        lstm_out_dim=len(X_test_n[0]),
    ).to(device)

    predicated_lstm = StockPricePredictor(
        short_numerical_dim=len(X_test_n[0]),
        numerical_dim=len(X_test_n[0]),
        lstm_out_dim=len(X_test_n[0]),
    ).to(device)

    long_estimated_lstm.load_state_dict(
        torch.load(path_long_model + args.long_estimated_model)
    )
    short_estimated_lstm.load_state_dict(
        torch.load(path_short_model + args.short_estimated_model)
    )
    predicated_lstm.load_state_dict(
        torch.load(path_longshort_model + args.predicted_model)
    )

    feats_n = prepare_data(
        seq_len,
        np.arange(seq_len, X_test_n.shape[0] + 1),
        torch.tensor(
            X_test_n.reshape(-1, len(X_test_n[0])), dtype=torch.float, device=device
        ),
        device,
    )

    feats_t_long = prepare_data(
        seq_len,
        np.arange(seq_len, X_test_n.shape[0] + 1),
        torch.tensor(X_test_t_long, dtype=torch.float, device=device),
        device,
    )

    feats_t_short = prepare_data(
        seq_len,
        np.arange(seq_len, X_test_n.shape[0] + 1),
        torch.tensor(X_test_t_short, dtype=torch.float, device=device),
        device,
    )

    _, out_long = long_estimated_lstm.forward(feats_t_long, feats_n)
    _, out_short = short_estimated_lstm.forward(feats_t_short, feats_n)
    y_hat = predicated_lstm.forward(out_long, out_short, feats_n)

    predict_price = (
        y_hat.view(-1).to("cpu").detach().numpy().reshape(-1, len(X_test_n[0]))
    )
    predict_price = scaler_y_test.inverse_transform(predict_price)

    result(
        predict_price,
        seq_len,
        path_long_pkl + args.num,
        path_data + args.stock,
        args.graph,
    )

    with open(path_longshort_pkl + args.output, "wb") as f:
        pkl.dump(predict_price, f)


if __name__ == "__main__":
    args = parser_args()
    test(args)
