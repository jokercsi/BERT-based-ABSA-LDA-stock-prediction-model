import argparse

import numpy as np
import torch
import torch.optim as optim
import tqdm
from LSTM import StockPriceEstimator, StockPricePredictor, data_load, prepare_data
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


path_pkl = "./../../data/pkl/long/"
path_model = "./../../data/model/long/"


def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num", default="stock_num.pkl")
    parser.add_argument("-t", "--text", default="stock_index_text")

    parser.add_argument("-e", "--epoch", default=1000, type=int)
    parser.add_argument("-b", "--batch_size", default=30, type=int)
    parser.add_argument("-s", "--seq_len", default=20, type=int)

    parser.add_argument("-e_model", "--estimated_model", default="estimated.model")

    parser.add_argument(
        "-p_model",
        "--predicted_model",
        default="predicted.model",
    )

    return parser.parse_args()


def estimate(
    X_train_n, X_train_t, y_train, seq_len, batch_size, epochs, path_long_model
):

    net = StockPriceEstimator(
        textual_dim=len(X_train_t[0]),
        numerical_dim=len(X_train_n[0]),
        dense_out_dim=int(len(X_train_t[0]) / 2),
        lstm_out_dim=len(X_train_n[0]),
    ).to(device)
    optimizer = optim.Adam(net.parameters())

    for _ in tqdm.tqdm(range(epochs)):
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

            y_target = torch.tensor(
                y_train[batch_idx - 1], dtype=torch.float, device=device
            )

            optimizer.zero_grad()
            y_hat, _ = net.forward(feats_t, feats_n)
            loss = (y_target - y_hat).pow(2).mean()
            loss.backward()
            optimizer.step()

    torch.save(net.state_dict(), path_long_model)


# LSTM 모델 학습
def train(args):

    scaler_x_train = MinMaxScaler()
    scaler_y_train = MinMaxScaler()
    scaler_x_val = MinMaxScaler()
    scaler_y_val = MinMaxScaler()

    # X_train_t 훈련 아마 토픽 데이터? 
    # X_train_n 훈련 가격 데이터 (Open)
    # y_train 훈련 가격 데이터 (Close)
    # X_val_t 검증 아마 토픽 데이터? 
    # X_val_n 검증 가격 데이터 (Open)
    # y_val 검증 가격 데이터 (Close) 
    # data_load 함수는 lstm 파일에 있다.
    X_train_t, X_train_n, y_train, X_val_t, X_val_n, y_val, _, _, _ = data_load(
        path_pkl + args.text, path_pkl + args.num
    )

    # 가격 데이터의 MinMaxScaler
    X_train_n = scaler_x_train.fit_transform(X_train_n)
    y_train = scaler_y_train.fit_transform(y_train)
    X_val_n = scaler_x_val.fit_transform(X_val_n)
    y_val = scaler_y_val.fit_transform(y_val)

    # LSTM 파일
    estimated_lstm = StockPriceEstimator(
        textual_dim=len(X_train_t[0]),
        numerical_dim=len(X_train_n[0]),
        dense_out_dim=int(len(X_train_t[0]) / 2),
        lstm_out_dim=len(X_train_n[0]),
    ).to(device)

    predicated_lstm = StockPricePredictor(
        numerical_dim=len(X_train_n[0]),
        lstm_out_dim=len(X_train_n[0]),
    ).to(device)

    optimizer = optim.Adam(predicated_lstm.parameters())

    min_loss = 999
    seq_len = args.seq_len
    epochs = args.epoch
    batch_size = args.batch_size

    # lstm 학습 결과가 저장 되는 곳
    # path_model + args.estimated_model = ./../../data/model/long/estimated.model
    estimate(
        X_train_n,
        X_train_t,
        y_train,
        seq_len,
        batch_size,
        epochs,
        path_model + args.estimated_model,
    )
    estimated_lstm.load_state_dict(torch.load(path_model + args.estimated_model))

    for _ in tqdm.tqdm(range(epochs)):
        idx = np.arange(seq_len, X_train_n.shape[0] + 1)
        tmp = 0

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

            y_target = torch.tensor(
                y_train[batch_idx - 1], dtype=torch.float, device=device
            )

            optimizer.zero_grad()

            y_hat = predicated_lstm.forward(out, feats_n)
            loss = (y_target - y_hat).pow(2).mean()
            tmp += loss
            loss.backward()
            optimizer.step()

        feats_n = prepare_data(
            seq_len,
            np.arange(seq_len, X_val_n.shape[0] + 1),
            torch.tensor(
                X_val_n.reshape(-1, len(X_val_n[0])), dtype=torch.float, device=device
            ),
            device,
        )

        feats_t = prepare_data(
            seq_len,
            np.arange(seq_len, X_val_n.shape[0] + 1),
            torch.tensor(X_val_t, dtype=torch.float, device=device),
            device,
        )
        _, out = estimated_lstm.forward(feats_t, feats_n)

        y_hat = predicated_lstm.forward(out, feats_n)
        pre = y_hat.view(-1).to("cpu").detach().numpy().reshape(-1, len(X_val_n[0]))
        tmp = 0
        tmp = np.mean((y_val[(seq_len - 1) :] - pre) ** 2)
        if tmp < min_loss:
            min_loss = tmp
            torch.save(predicated_lstm.state_dict(), path_model + args.predicted_model)


if __name__ == "__main__":
    args = parser_args()
    
    train(args)
    # print(path_model + args.estimated_model) # ./../../data/model/long/estimated.model
    # torch.load: pickle을 사용하여 저장된 객체 파일들을 역직렬화하여 메모리에 올립니다.
    # print(torch.load(path_model + args.estimated_model))
