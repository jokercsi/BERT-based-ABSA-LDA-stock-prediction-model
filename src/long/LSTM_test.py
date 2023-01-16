import argparse
import os
import pathlib
import pickle as pkl

import numpy as np
import torch
from LSTM import StockPriceEstimator, StockPricePredictor, data_load, prepare_data
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

path_data = "./../../data/"
path_pkl = "./../../data/pkl/long/"
path_model = "./../../data/model/long/"


def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num", default="stock_num.pkl")
    parser.add_argument("-t", "--text", default="stock_index_text")
    parser.add_argument("-stock", "--stock", default="price")
    parser.add_argument("-graph", "--graph", default="./../../data/result/long")

    parser.add_argument("-s", "--seq_len", default=20, type=int)    # 학습기간

    parser.add_argument("-e_model", "--estimated_model", default="estimated.model")
    parser.add_argument(
        "-p_model",
        "--predicted_model",
        default="predicted.model",
    )
    parser.add_argument("-o", "--output", default="predict_stock.pkl")

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
        num_data, shuffle=False, test_size=None, train_size=454
    )
    num_val, num_test = train_test_split(
        num_test, shuffle=False, test_size=None, train_size=97
    )

    open_price = num_test[:, 0::4]
    low_price = num_test[:, 1::4]
    high_price = num_test[:, 2::4]
    close_price = num_test[:, 3::4]
    # print( open_price, low_price, high_price, close_price)
    # [[26774.61914062 10500.51953125  3224.20996094]]
    # [[26619.88085938 10421.20996094  3205.64990234]]
    # [[26808.4296875  10532.62011719  3233.52001953]]
    # [[26671.94921875 10503.19042969  3224.72998047]]

    return open_price, low_price, high_price, close_price


def mcfd(predict_price, close_price):

    # print(predict_price, close_price) # close_price가 0....

    predict_future = predict_price[1:, :]
    predict_current = predict_price[:-1, :]
    predict = np.log(predict_future) - np.log(predict_current)

    label_future = close_price[1:, :]
    label_current = close_price[:-1, :]
    label = np.log(label_future) - np.log(label_current)

    tmp = np.zeros_like(predict)
    mcfd = np.zeros((1, tmp.shape[1]))

    for i in range(tmp.shape[1]):
        for j in range(tmp.shape[0]):
            if predict[j, i] * label[j, i] >= 0.0:
                tmp[j, i] = 1
        mcfd[0, i] = np.mean(tmp[:, i])

    return mcfd[0]


def mftr(predict_price, close_price):

    predict_future = predict_price[1:, :]
    predict_current = predict_price[:-1, :]
    predict = np.log(predict_future) - np.log(predict_current)

    label_future = close_price[1:, :]
    label_current = close_price[:-1, :]
    label = np.log(label_future) - np.log(label_current)

    tmp = np.zeros_like(predict)
    mftr = np.zeros((1, tmp.shape[1]))
    for i in range(tmp.shape[1]):
        for j in range(tmp.shape[0]):
            tmp[j, i] = np.sign(predict[j, i]) * label[j, i]
        mftr[0, i] = np.mean(tmp[:, i])
    return mftr[0]


def trade(predict_price, open_price, low_price, high_price, close_price):

    predict_change = (predict_price - open_price) / open_price
    returns = np.zeros(predict_change.shape)
    fees = 20

    for i in range(predict_change.shape[1]):
        for j in range(predict_change.shape[0]):
            if predict_change[j, i] >= 0:
                if high_price[j, i] >= open_price[j, i] * 1.02:
                    returns[j, i] += (
                        (open_price[j, i] * 0.02) * (1000000 / open_price[j, i])
                    ) - fees
                else:
                    returns[j, i] += (
                        (-open_price[j, i] + close_price[j, i])
                        * (1000000 / open_price[j, i])
                    ) - fees

            elif predict_change[j, i] < 0:
                if low_price[j, i] <= open_price[j, i] * 0.98:
                    returns[j, i] += (
                        (open_price[j, i] * 0.02) * (1000000 / open_price[j, i])
                    ) - fees
                else:
                    returns[j, i] += (
                        (open_price[j, i] - close_price[j, i])
                        * (1000000 / open_price[j, i])
                    ) - fees

    return sum(returns)


def result(predict_price, seq_len, path_num, path_stock, path_graph):
    company_id = read_company_id(path_stock)
    path_graph = pathlib.Path(path_graph)

    # test_data_load 함수
    open_price, low_price, high_price, close_price = test_data_load(path_num)

    #print(open_price, low_price, high_price, close_price)

    open_price = open_price[(seq_len - 1) :]
    low_price = low_price[(seq_len - 1) :]
    high_price = high_price[(seq_len - 1) :]
    close_price = close_price[(seq_len - 1) :]

    #print(open_price, low_price, high_price, close_price)

    mcfd_result = mcfd(predict_price, close_price)  # mcfd 함수 가져오기
    mftr_result = mftr(predict_price, close_price)  # mftr 함수 가져오기
    
    # trade 함수 가져오기
    trade_result = trade(predict_price, open_price, low_price, high_price, close_price)

    for i, (company, mcfd_, mftr_, trade_) in enumerate(
        zip(company_id, mcfd_result, mftr_result, trade_result)
    ):
        print("--------------" + company + "-----------------")
        print("mcfd: ", mcfd_, "mftr:", mftr_, " returns: ", trade_)

        plt.figure(figsize=(15, 8), dpi=300)
        plt.plot(predict_price[:, i], label="predict_price")
        # plt.plot(open_price[:, i], label="open_price")
        plt.plot(close_price[:, i], label="close_price")
        # plt.legend(["predict_price", "open_price", "close_price"], fontsize=20)
        plt.legend(["predict_price", "close_price"], fontsize=20)

        file_name = company + ".png"
        plt.savefig(path_graph / file_name, dpi=300)


# MAIN 함수에서 불러 들이키는 함수
def test(args):

    scaler_x_test = MinMaxScaler()
    scaler_y_test = MinMaxScaler()

    _, _, _, _, _, _, X_test_t, X_test_n, y_test = data_load(
        path_pkl + args.text, path_pkl + args.num
    )
    # print(X_test_n, y_test)
    # [[26774.61914062 10500.51953125  3224.20996094]]
    # [[26671.94921875 10503.19042969  3224.72998047]]
    X_test_n = scaler_x_test.fit_transform(X_test_n)
    y_test = scaler_y_test.fit_transform(y_test)

    # print(X_test_n, y_test) # LFTM

    estimated_lstm = StockPriceEstimator(
        textual_dim=len(X_test_t[0]),
        numerical_dim=len(X_test_n[0]),
        dense_out_dim=int(len(X_test_t[0]) / 2),
        lstm_out_dim=len(X_test_n[0]),
    ).to(device)

    predicated_lstm = StockPricePredictor(
        numerical_dim=len(X_test_n[0]),
        lstm_out_dim=len(X_test_n[0]),
    ).to(device)

    estimated_lstm.load_state_dict(torch.load(path_model + args.estimated_model))
    predicated_lstm.load_state_dict(torch.load(path_model + args.predicted_model))

    seq_len = args.seq_len

    feats_n = prepare_data(
        seq_len,
        np.arange(seq_len, X_test_n.shape[0] + 1),
        torch.tensor(
            X_test_n.reshape(-1, len(X_test_n[0])), dtype=torch.float, device=device
        ),
        device,
    )

    feats_t = prepare_data(
        seq_len,
        np.arange(seq_len, X_test_n.shape[0] + 1),
        torch.tensor(X_test_t, dtype=torch.float, device=device),
        device,
    )

    _, out = estimated_lstm.forward(feats_t, feats_n)
    predict_price = predicated_lstm.forward(out, feats_n)
    predict_price = (
        predict_price.view(-1).to("cpu").detach().numpy().reshape(-1, len(X_test_n[0]))
    )
    predict_price = scaler_y_test.inverse_transform(predict_price)

    # 결과 함수 가져오기
    result(
        predict_price, seq_len, path_pkl + args.num, path_data + args.stock, args.graph
    )

    with open(path_pkl + args.output, "wb") as f:
        pkl.dump(predict_price, f)


if __name__ == "__main__":
    args = parser_args()
    test(args)
