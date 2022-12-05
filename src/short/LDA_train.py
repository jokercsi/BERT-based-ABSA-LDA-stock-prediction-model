import argparse
import datetime
import pathlib

import pandas as pd
from dateutil.relativedelta import relativedelta
from LDA import LDA_est
from monthdelta import monthmod

start = datetime.date(2015, 1, 1)
end = datetime.date(2018, 7, 1)


def parser_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-lda", "--lda", default="./../../lda/src", help="lda/srcまでの相対パス"
    )
    parser.add_argument("-stock", "--stock", default="./../../data/TOPIX10C_CAR")
    parser.add_argument(
        "-train",
        "--train",
        default="./../../data/train/",
        help="lda/srcから推定データまでの相対パス",
    )
    parser.add_argument(
        "-current",
        "--current",
        default="./../../src/short/",
        help="lda/srcから現在のフォルダまでの相対パス",
    )

    parser.add_argument("-m", "--month", default=12)
    parser.add_argument("-s", "--seq_len", default=20)

    parser.add_argument("-alpha", "--alpha", default=0.1)
    parser.add_argument("-beta", "--beta", default=0.1)
    parser.add_argument("-ntopics", "--topics", default=50)
    parser.add_argument("-niters", "--niters", default=1000)
    parser.add_argument("-savesteps", "--savesteps", default=500)
    parser.add_argument("-twords", "--twords", default=100)

    return parser.parse_args()


def read_csv(path_stock):

    file_names = path_stock.glob("*.csv")

    date = []
    for f in file_names:
        stock_df = pd.read_csv(f, encoding="CP932")
        if len(stock_df) == 1219:
            date.extend([int(t.replace("-", "")) for t in stock_df["日付"]])
    date_list = sorted(list(set(date)))

    date_list = pd.to_datetime(date_list, format="%Y%m%d")

    return date_list


if __name__ == "__main__":
    args = parser_args()

    path_stock = pathlib.Path(args.stock)

    date_list = read_csv(path_stock)

    month_datetime = relativedelta(months=int(args.month))
    month = int(args.month)
    seq_len = args.seq_len - 2

    date_start = start
    date_to = date_start + month_datetime

    for _ in range(int(monthmod(start, end)[0].months / month)):

        date_end = list(filter(lambda x: x > date_to, date_list.date))[seq_len]
        if monthmod(date_to, end)[0].months < month:
            date_end = end - relativedelta(days=1)

        term = str(date_start) + "_" + str(date_end)
        path_train = args.train + term + "/train.txt"

        LDA_est(
            args.lda,
            args.alpha,
            args.beta,
            args.topics,
            args.niters,
            args.savesteps,
            args.twords,
            path_train,
            args.current,
        )

        date_start += month_datetime
        date_to += month_datetime
