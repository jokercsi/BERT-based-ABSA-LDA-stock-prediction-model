import argparse
import datetime
import os
import pathlib

import pandas as pd
from dateutil.relativedelta import relativedelta
from monthdelta import monthmod
from sklearn.model_selection import train_test_split

start = datetime.date(2015, 1, 1)
end = datetime.date(2018, 7, 1)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="./../../data/Morphological_data.csv")
    parser.add_argument("-stock", "--stock", default="./../../data/TOPIX10C_CAR")
    parser.add_argument("-train", "--train", default="./../../data/train/")
    parser.add_argument("-test", "--test", default="./../../data/test/")

    parser.add_argument("-m", "--month", default=12)
    parser.add_argument("-s", "--seq_len", default=20)

    return parser.parse_args()


def read_csv(path_news, path_stock):

    news_df = pd.read_csv(path_news, encoding="utf8")
    news_df["掲載日"] = pd.to_datetime(news_df["掲載日"], format="%Y%m%d")
    file_names = path_stock.glob("*.csv")

    date = []
    for f in file_names:
        stock_df = pd.read_csv(f, encoding="CP932")
        if len(stock_df) == 1219:
            date.extend([int(t.replace("-", "")) for t in stock_df["日付"]])
    date_list = sorted(list(set(date)))

    date_list = pd.to_datetime(date_list, format="%Y%m%d")

    return news_df, date_list


def make_data(data, path):

    if not os.path.exists(path):
        path.mkdir()

    path = path / "train.txt"
    file_out = open(path, "w", encoding="utf_8")
    file_out.write(str(len(data)))
    file_out.write("\n")
    for i in data.index:
        file_out.write(data["本文"][i])
        file_out.write("\n")
    file_out.close()


def split_data(args):

    path_news = pathlib.Path(args.input)
    path_stock = pathlib.Path(args.stock)

    news_df, date_list = read_csv(path_news, path_stock)

    month_datetime = relativedelta(months=int(args.month))
    month = int(args.month)
    seq_len = args.seq_len - 2

    path_train = pathlib.Path(args.train)
    # path_test = pathlib.Path(args.test)

    date_start = start
    date_to = date_start + month_datetime

    for _ in range(int(monthmod(start, end)[0].months / month)):

        date_end = list(filter(lambda x: x > date_to, date_list.date))[seq_len]
        if monthmod(date_to, end)[0].months < month:
            date_end = end - relativedelta(days=1)

        df = news_df[
            (str(date_start) <= news_df["掲載日"]) & (news_df["掲載日"] <= str(date_end))
        ]
        train, test = train_test_split(
            df, shuffle=False, test_size=None, train_size=0.7
        )

        term = str(date_start) + "_" + str(date_end)
        make_data(train, path_train / term)
        #         make_data(test , path_test / term)

        date_start += month_datetime
        date_to += month_datetime


if __name__ == "__main__":
    args = parser_args()
    split_data(args)
