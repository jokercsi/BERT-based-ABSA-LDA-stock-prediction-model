import argparse
import pathlib
import pickle as pkl

import numpy as np
import pandas as pd

path_data = "./../../data/"
path_pkl = "./../../data/pkl/long/"


def parser_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-stock", "--stock", default="price")
    parser.add_argument("-o", "--output", default="stock_num.pkl")

    return parser.parse_args()


def read_csv(args):

    path_csv = pathlib.Path(path_data + args.stock)
    path_output = pathlib.Path(path_pkl + args.output)

    stock_data = []
    date = []
    file_names = path_csv.glob("*.csv")

    for f in file_names:
        #print(f)
        stock_df = pd.read_csv(f, encoding="CP932")
        print(stock_df)
        if len(stock_df) == 649:
            date.extend(list(stock_df["Date"]))
            #date.extend([int(t.replace("/", "")) for t in stock_df["Date"]])

    #print(date)
    date_list = set(date)
    file_names = path_csv.glob("*.csv")

    for f in file_names:
        df = pd.read_csv(f, encoding="CP932")
        print(df["Date"].isin(date_list))
        tmp = df[df["Date"].isin(date_list)][["Open", "Low", "High", "Close"]].values.reshape(-1, 4)
        stock_data.append(tmp)

    print(stock_data)
    stock_data = np.concatenate(stock_data, axis=1)
    print(stock_data)

    output(stock_data, path_output)


def output(data, path_output):
    with open(path_output, "wb") as f:
        pkl.dump(data, f)


if __name__ == "__main__":
    args = parser_args()
    read_csv(args)
