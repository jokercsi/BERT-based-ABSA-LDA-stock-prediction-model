import argparse
import pathlib
import pickle as pkl

import numpy as np
import pandas as pd

path_data = "./../../data/"
path_pkl = "./../../data/pkl/long/"


def parser_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-stock", "--stock", default="TOPIX10C_CAR")
    parser.add_argument("-o", "--output", default="car_num.pkl")

    return parser.parse_args()


def read_csv(args):

    path_csv = pathlib.Path(path_data + args.stock)
    path_output = pathlib.Path(path_pkl + args.output)

    stock_data = []
    date = []
    file_names = path_csv.glob("*.csv")

    for f in file_names:
        df = pd.read_csv(f, encoding="CP932")
        if len(df) == 1219:
            date.extend(list(df["日付"]))

    date_list = set(date)
    file_names = path_csv.glob("*.csv")

    for f in file_names:
        df = pd.read_csv(f, encoding="CP932")
        tmp = df[df["日付"].isin(date_list)][["始値", "安値", "高値", "終値"]].values.reshape(
            -1, 4
        )
        stock_data.append(tmp)

    stock_data = np.concatenate(stock_data, axis=1)
    output(stock_data, path_output)


def output(data, path_output):
    with open(path_output, "wb") as f:
        pkl.dump(data, f)


if __name__ == "__main__":
    args = parser_args()
    read_csv(args)
