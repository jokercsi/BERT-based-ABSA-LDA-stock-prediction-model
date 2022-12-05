import argparse
import datetime
import os
import pathlib
import pickle as pkl
import warnings

import numpy as np
import pandas as pd
import tqdm
from dateutil.relativedelta import relativedelta
from LDA import LDA_inf
from monthdelta import monthmod

warnings.filterwarnings("ignore")

start = datetime.date(2015, 1, 1)
end = datetime.date(2018, 7, 1)

path_data = "./../../data/"
path_pkl = "./../../data/pkl/short/"


def parser_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--news", default="./../../data/Morphological_data.csv")
    parser.add_argument("-list", "--company_list", default="car_list.txt")
    parser.add_argument("-stock", "--stock", default="TOPIX10C_CAR")

    parser.add_argument(
        "-lda", "--lda", default="./../../lda/src", help="現在のフォルダからlda/srcまでの相対パス"
    )
    parser.add_argument(
        "-train",
        "--train",
        default="./../../data/train/",
        help="lda/srcから学習済みLDAモデルがあるフォルダまでの相対パス",
    )
    parser.add_argument(
        "-test",
        "--test",
        default="./../../vector/",
        help="学習済みLDAモデルがあるフォルダから推定データの相対パス",
    )
    parser.add_argument(
        "-vector",
        "--vector",
        default="./../../data/vector/",
        help="現在のフォルダから日付ごとに新聞記事を作成するフォルダまでの相対パス",
    )
    parser.add_argument(
        "-current",
        "--current",
        default="./../../src/short/",
        help="lda/srcから現在のフォルダまでの相対パス",
    )

    parser.add_argument("-niters", "--niters", default=30)
    parser.add_argument("-twords", "--twords", default=100)
    parser.add_argument("-topics", "--topics", default=50)

    parser.add_argument("-o", "--output", default="car_text.pkl")

    parser.add_argument("-m", "--month", default=12)
    parser.add_argument("-s", "--seq_len", default=20)

    return parser.parse_args()


def read_company_id(path_stock):

    id_name = os.listdir(path_stock)
    name = [
        f.replace(".csv", "")
        for f in id_name
        if os.path.isfile(os.path.join(path_stock, f))
    ]

    return name


def read_company_list(path_company_list):
    with open(path_company_list, "r", encoding="utf-8") as f:
        company = [(w.replace("\n", "")).split(",") for w in f]
        f.close()
    return company


def read_csv(path_news, path_stock):

    news_df = pd.read_csv(path_news, encoding="utf8")
    file_names = path_stock.glob("*.csv")

    date = []
    for f in file_names:
        stock_df = pd.read_csv(f, encoding="CP932")
        if len(stock_df) == 1219:
            date.extend([int(t.replace("-", "")) for t in stock_df["日付"]])
    date_list = sorted(list(set(date)))

    return news_df, date_list


def check_company_noun(df, company_list):
    text = str(df[8]) + " " + df[9]

    noun = text.split(" ")
    noun = list(filter(("").__ne__, noun))
    check_list = []
    for i, cl in enumerate(company_list):
        for c in cl:
            if c in noun:
                check_list.append(i)
                break

    return check_list


def extract_news(company_list, date_list, news_df):
    company_index_dict = dict()

    for date in tqdm.tqdm(date_list):
        company_index_list = [[] for i in range(len(company_list))]

        for d, i in zip(
            news_df[news_df["掲載日"] == date].values,
            news_df[news_df["掲載日"] == date].index,
        ):

            relation_list = check_company_noun(d, company_list)

            if relation_list != []:
                for j in relation_list:
                    company_index_list[j].append(i)
        company_index_dict[date] = company_index_list
    return company_index_dict


def make_folder(data_path, date_list, company_index_dict, company_list, news, term):

    data_path = data_path / term

    if not os.path.exists(data_path):
        data_path.mkdir()

    for i, date in enumerate(date_list):
        path = data_path / str(date)

        if not os.path.exists(path):
            path.mkdir()

        for j, v in enumerate(company_index_dict[date]):
            if v != []:
                text = []
                for idx in v:
                    text.append(news["本文"][idx])

                if not os.path.exists(path / company_list[j]):
                    (path / company_list[j]).mkdir()

                file_out = open(
                    path / company_list[j] / "vector.txt", "w", encoding="utf_8"
                )
                file_out.write("1")
                file_out.write("\n")
                sentense = ""
                for i in range(0, len(text)):
                    sentense += text[i] + " "
                file_out.write(sentense)
                file_out.write("\n")
                file_out.close()


def LDA(args, company_id, date_list, company_index_dict, term, term_test):

    path_lda = pathlib.Path(args.lda)
    train = args.train + term + "/"
    test = args.test + term_test + "/"
    path_vector = pathlib.Path(args.vector) / term_test
    path_return = pathlib.Path(args.current)

    niters = args.niters
    twords = args.twords
    topics = args.topics
    model = "model-final"

    data = []

    for date in date_list:

        topic_vector = np.zeros((len(company_id), int(topics)))
        path = path_vector / str(date)

        for j, v in enumerate(company_index_dict[date]):
            if v != []:

                tests = test + str(date) + "/" + company_id[j] + "/vector.txt"
                LDA_inf(path_lda, train, model, niters, twords, tests, path_return)
                theta = []

                with open(path / company_id[j] / "vector.txt.theta", "r") as fin:
                    for line in fin.readlines():
                        row = []
                        toks = line.split(" ")
                        for tok in toks:
                            try:
                                tok = float(tok)
                            except ValueError:
                                continue

                            row.append(tok)
                        theta.append(row)
                theta = np.array(theta)
                topic_vector[j] = theta

        tmp = np.concatenate(topic_vector).reshape(1, int(topics) * len(company_id))
        data.append(tmp)
    data = np.concatenate(data)

    return data


#     output(data, output_path)


def output(data, path_output):
    with open(path_output, "wb") as f:
        pkl.dump(data, f)


if __name__ == "__main__":

    args = parser_args()
    path_news = pathlib.Path(args.news)
    path_stock = pathlib.Path(path_data + args.stock)
    path_vector = pathlib.Path(args.vector)
    path_list = pathlib.Path(path_data + args.company_list)
    path_output = pathlib.Path(path_pkl + args.output)

    month_datetime = relativedelta(months=int(args.month))
    month = int(args.month)
    seq_len = args.seq_len - 2

    company_id = read_company_id(path_stock)
    company_list = read_company_list(path_list)

    news, date_list = read_csv(path_news, path_stock)
    company_index_dict = extract_news(company_list, date_list, news)

    date_start = start
    date_to = date_start + month_datetime
    date_list = pd.to_datetime(date_list, format="%Y%m%d")

    topic_vector = []

    for _ in range(int(monthmod(start, end)[0].months / month)):

        date_end = list(filter(lambda x: x > date_to, date_list.date))[seq_len]

        if monthmod(date_to, end)[0].months < month:
            date_end = end - relativedelta(days=1)

        date_list_term = date_list[
            (str(date_start) <= date_list) & (date_list <= str(date_end))
        ].date
        date_list_term = [int(str(date).replace("-", "")) for date in date_list_term]

        term = str(date_start) + "_" + str(date_end)

        make_folder(
            path_vector, date_list_term, company_index_dict, company_id, news, term
        )
        topic_vector.append(
            LDA(args, company_id, date_list_term, company_index_dict, term, term)
        )

        date_start += month_datetime
        date_to += month_datetime

    date_list_term = date_list[str(end) <= date_list].date
    date_list_term = [int(str(date).replace("-", "")) for date in date_list_term]

    term_test = "test" + str(month) + "month"

    make_folder(
        path_vector, date_list_term, company_index_dict, company_id, news, term_test
    )
    topic_vector.append(
        LDA(args, company_id, date_list_term, company_index_dict, term, term_test)
    )

    output(topic_vector, path_output)
