import argparse
import os
import pathlib
import pickle as pkl
import warnings

import numpy as np
import pandas as pd
import tqdm
from LDA import LDA_inf

warnings.filterwarnings("ignore")

path_data = "./../../data/"
path_pkl = "./../../data/pkl/long/"


def parser_args():

    # 현재 위치 확인
    cwd = os.getcwd()
    print("current location is :", cwd)

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--news", default="./../../data/output_morphological.csv")

    parser.add_argument("-list", "--company_list", default="stock_index.txt")

    parser.add_argument("-stock", "--stock", default="price")

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
        default="./../vector/",
        help="学習済みLDAモデルがあるフォルダから推定データの相対パス",
    )
    parser.add_argument(
        "-vector",
        "--vector",
        default="./../../data/vector",
        help="現在のフォルダから日付ごとに新聞記事を作成するフォルダまでの相対パス",
    )
    parser.add_argument(
        "-current",
        "--current",
        default="./../../src/long/",
        help="lda/srcから現在のフォルダまでの相対パス",
    )

    parser.add_argument("-niters", "--niters", default=30)
    parser.add_argument("-twords", "--twords", default=100)
    parser.add_argument("-topics", "--topics", default=100)

    parser.add_argument("-o", "--output", default="car_text.pkl")

    return parser.parse_args()


def read_company_id(path_stock):

    id_name = os.listdir(path_stock)
    name = [
        f.replace(".csv", "")
        for f in id_name
        if os.path.isfile(os.path.join(path_stock, f))
    ]

    return name


# stock_index 내용 읽기
def read_company_list(path_company_list):
    with open(path_company_list, "r", encoding="utf-8") as f:
        company = [(w.replace("\n", "")).split(",") for w in f]
        f.close()

        # print(company)
    return company


# 가격 데이터 읽는 함수
def read_csv(path_news, path_stock):

    news_df = pd.read_csv(path_news, encoding="utf-8")    # news_df : output_morpological을 읽음
    file_names = path_stock.glob("*.csv")

    date = []
    for f in file_names:    # f는 파일 경로
        stock_df = pd.read_csv(f, encoding="utf-8")  # df 가격 파일 데이터 가져오기
        if len(stock_df) == 649:
            date.extend([int(t.replace("/", "")) for t in stock_df["Date"]])

    date_list = sorted(list(set(date)))  # 가격 데이터의 날짜 모두 가져오기
    #  print(date_list)

    return news_df, date_list


# list 에 있는 회사 뽑아옴
def check_company_noun(news, company_list):
    text = news[4]
    # text = 'x a x'
    noun = text.split(" ")
    noun = list(filter(("").__ne__, noun))  # 뉴스 기사를 단어 단위로 나눔
    check_list = []
    # print(noun)

    for i, company_name in enumerate(company_list):
        for name in company_name:
            if name in noun:
                check_list.append(i)
                break
    return check_list


def extract_news(company_list, date_list, news_df):
    company_index_dict = dict()

    # 날짜 데이터 형식 바꾸기
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    news_df['Date'] = news_df["Date"].dt.strftime('%Y%m%d')
    news_df['Date'] = news_df['Date'].astype(int)
    # print(news_df['Date'])

    for date in tqdm.tqdm(date_list):   # 가격의 날짜
        # len(company_list) : 3 (Dow, SNP, Nasdaq)
        company_index_list = [[] for i in range(len(company_list))]
        # print(type(news_df["Date"]))
        # print(type(date))
        # print(news_df[news_df["Date"] == date].values)
        for news, index in zip(
            news_df[news_df["Date"] == date].values,
            news_df[news_df["Date"] == date].index,
        ):

            relation_list = check_company_noun(news, company_list)
            # print(relation_list)

            if relation_list != []:
                for j in relation_list:
                    company_index_list[j].append(index)
        company_index_dict[date] = company_index_list
    return company_index_dict


# 뉴스 기사에 stock_index 리스트의 단어가 있으면 폴더 만들기
def make_folder(data_path, date_list, company_index_dict, company_list, news):

    for i, date in enumerate(date_list):

        path = data_path / str(date)

        if not os.path.exists(path):
            path.mkdir()

        for j, v in enumerate(company_index_dict[date]):
            if v != []:
                text = []
                for idx in v:
                    text.append(news["after_headlines"][idx])

                if not os.path.exists(path / company_list[j]):
                    (path / company_list[j]).mkdir()

                # vector.txt 만들기;
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


def LDA(args, company_id, date_list, company_index_dict):

    path_lda = pathlib.Path(args.lda)   # ./lda/src
    train = args.train
    test = args.test
    path_return = os.getcwd()

    niters = args.niters
    twords = args.twords
    topics = args.topics
    model = "model-final"
    print(args)
    topic_vector = []

    for date in date_list:

        topic_vector_data = np.zeros((len(company_id), int(topics)))
        path = path_vector / str(date)

        for j, v in enumerate(company_index_dict[date]):
            if v != []:

                # 현재 위치 확인
                cwd = os.getcwd()
                print("current location is :", cwd)
                print(test)
                # ./../vector/20171227/SnP/vector.txt
                tests = test + str(date) + "/" + company_id[j] + "/vector.txt"
                LDA_inf(path_lda, train, model, niters, twords, tests, path_return)
                theta = []
                # 토픽을 백터화
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
                topic_vector_data[j] = theta

        tmp = np.concatenate(topic_vector_data).reshape(
            1, int(topics) * len(company_id)
        )
        topic_vector.append(tmp)
    topic_vector = np.concatenate(topic_vector)
    output(topic_vector, path_pkl + args.output)


def output(data, path_output):
    with open(path_output, "wb") as f:
        pkl.dump(data, f)


if __name__ == "__main__":

    args = parser_args()
    path_news = pathlib.Path(args.news)                     # data\output_morphological.csv
    path_stock = pathlib.Path(path_data + args.stock)       # data\price
    path_vector = pathlib.Path(args.vector)                 # data\vector
    path_list = pathlib.Path(path_data + args.company_list)  # data\stock_index.txt

    company_id = read_company_id(path_stock)                # ['Dow', 'Nasdaq', 'SnP']
    # [['S&P500', ' S&P', ' s&p500', ' GSPC', ' SPX'],
    #  ['NASDAQ', ' Nasdaq', ' IXIC'], ['Dow', ' DJI']]
    company_list = read_company_list(path_list)
    # news : 데이터 마이닝 후의 텍스트 데이터  , date_list : 가격 데이터의 날짜
    news, date_list = read_csv(path_news, path_stock)
    # stock_index 리스트에 있는 단어가 포함된 뉴스만 뽑아오기
    company_index_dict = extract_news(company_list, date_list, news)
    # print(company_index_dict)
    # 폴더 만들기 (data\vector, 가격 데이터의 날짜,
    # stock_index의 단어가 포함된 뉴스, ['Dow', 'Nasdaq', 'SnP'], 데이터 마이닝 후의 텍스트 데이터)
    make_folder(path_vector, date_list, company_index_dict, company_id, news)
    LDA(args, company_id, date_list, company_index_dict)
