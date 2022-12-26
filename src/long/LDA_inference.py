import argparse
import os
import pathlib
import pickle as pkl
import warnings


import numpy as np
import pandas as pd
import tqdm
from LDA import LDA_inf
# from PyABSA_inference import preprocessing_pyABSA
# from PyABSA_inference import ABSA
# from itertools import chain


warnings.filterwarnings("ignore")

path_data = "./../../data/"
path_pkl = "./../../data/pkl/long/"

def parser_args():

    # 현재 위치 확인
    # cwd = os.getcwd()
    # print("current location is :", cwd)

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--news", default="./../../data/output_morphological.csv")

    parser.add_argument("-list", "--company_list", default="stock_index")

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

    parser.add_argument("-niters", "--niters", default=10)   # The number of Gibbs sampling iterations
    parser.add_argument("-twords", "--twords", default=10)  # The number of most likely words for each topic
    parser.add_argument("-topics", "--topics", default=10)  # The number of topics

    parser.add_argument("-o", "--output", default="stock_index_text")

    return parser.parse_args()


def read_company_id(path_stock):

    id_name = os.listdir(path_stock)
    name = [
        f.replace(".csv", "")
        for f in id_name
        if os.path.isfile(os.path.join(path_stock, f))
    ]

    return name


# stock_index 폴더의 내용 읽기
def read_company_list(path_company_list):

    file_names = path_company_list.glob("*.txt")

    stock_index = []
    for file_path in file_names:    # file_path는 파일 경로
        with open(file_path, "r", encoding="utf-8") as f:
            company_name = [w.strip() for w in f]
            company_name = list(company_name)
            stock_index.append(company_name)
            f.close()

    return stock_index


# 뉴스 데이터 읽는 함수
def read_csv(path_news, path_stock):

    news_df = pd.read_csv(path_news, encoding="utf_8")    # news_df : output_morpological을 읽음
    file_names = path_stock.glob("*.csv")

    date = []
    for f in file_names:    # f는 파일 경로
        stock_df = pd.read_csv(f, encoding="utf_8")  # df 가격 파일 데이터 가져오기
        if len(stock_df) == 649:
            date.extend([int(t.replace("/", "")) for t in stock_df["Date"]])

    date_list = sorted(list(set(date)))  # 가격 데이터의 날짜 모두 가져오기
    # print(date_list)

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

        # 날짜 폴더 만드는 코드 
        if not os.path.exists(path):
            path.mkdir()

        for j, v in enumerate(company_index_dict[date]):
            if v != []:
                text = []
                for idx in v:
                    text.append(news["after_headlines"][idx])

                # 회사 폴더 만드는 코드
                if not os.path.exists(path / company_list[j]):
                    (path / company_list[j]).mkdir()

                # vector.txt 만들기;
                file_out = open(
                    path / company_list[j] / "vector.txt", "w", encoding="utf_8"
                )
                file_out.write("1")  # 왜 1 일까??
                file_out.write("\n")
                sentense = ""
                for i in range(0, len(text)):
                    sentense += text[i] + " "
                file_out.write(sentense)
                file_out.write("\n")
                file_out.close()


def LDA(args, company_id, date_list, company_index_dict):
    # company_id : ['Dow', 'Nasdaq', 'SnP'] 3개
    
    path_lda = pathlib.Path(args.lda)  # lda의 위치 : ./lda/src
    train = args.train
    test = args.test
    path_return = os.getcwd()

    # LDA model args
    niters = args.niters
    twords = args.twords
    topics = args.topics
    model = "model-final"  # The name of the previously estimated model.

    topic_vector = []
    #for date in date_list:
    for date in date_list:
        #print("date =", date)

        # 3 (주가지수 수) X 10 (토픽 수)
        # 3 X 10 배열은 모두 0
        topic_vector_data = np.zeros((len(company_id), int(topics)))
        
        #print("topic_vector_data 3X10", topic_vector_data)
        path = path_vector / str(date)

        for j, v in enumerate(company_index_dict[date]):
            if v != []:  # 기사에 해당되는 내용이 있을 경우
                # tests =  ./../vector/20171227/SnP/vector.txt
                tests = test + str(date) + "/" + company_id[j] + "/vector.txt"
                # LDA Model Execute
                LDA_inf(path_lda, train, model, niters, twords, tests, path_return)

        #         theta = []
        #         # 토픽을 백터화 topic vector
        #         with open(path / company_id[j] / "vector.txt.theta", "r") as fin:
        #             for line in fin.readlines():
        #                 row = []
        #                 toks = line.split(" ")
        #                 for tok in toks:
        #                     #print(tok)
        #                     try:
        #                         tok = float(tok)
        #                     except ValueError:
        #                         continue

        #                     row.append(tok)
        #                 theta.append(row)
        #         #print("theta", theta)
        #         theta = np.array(theta)
        #         print("theta", theta)
        #         topic_vector_data[j] = theta
        # #print("topic_vector_data", topic_vector_data)
        # tmp = np.concatenate(topic_vector_data).reshape(
        #     1, int(topics) * len(company_id)
        # )
        # #print("tmp", tmp)
        # topic_vector.append(tmp)
    
    # print("topic_vector", topic_vector)
    # topic_vector = np.concatenate(topic_vector)
    # print("topic_vector", topic_vector)
    # output(topic_vector, path_pkl + args.output)

# LSTM을 위한 함수
def output(data, path_output):
    #print(data, path_output)
    with open(path_output, "wb") as f:
        pkl.dump(data, f)


def sentiment_analysis(path_sentiment):
    df = pd.read_csv(path_sentiment)
    df = df.sort_values('Date')
    neutral = df[df['Sentiment'].str.contains('Neutra')].index
    df.drop(neutral, inplace=True)
    df.to_csv("./../../data/sentiment/out_put_sentiment.csv", index=False, encoding="utf-8")
    return df


def replace_topic(args, date_list, company_index_dict, company_id, sentiment_df):    
    
    topics = args.topics
    topic_vector = []
    #for date in date_list:
    for date in date_list:
        path = path_vector / str(date)
        topic_vector_data = np.zeros((len(company_id), int(topics)))
        #print(topic_vector_data)
        for j, v in enumerate(company_index_dict[date]):
            if v != []:  # 기사에 해당되는 내용이 있을 경우

                twords = []
                # 토픽 단어 가져오기
                print(path / company_id[j] / "vector.txt.twords")
                with open(path / company_id[j] / "vector.txt.twords", "r", encoding='utf_8', errors='ignore') as fin:
                    topic_rank = 0
                    word_list = []
                    for line in fin.readlines():
                        #print(line)
                        toks = line.split("  ")
                        if len(toks) == 1:
                            topic_rank += 1
                            word_list = []
                        else:
                            word = toks[0][1:]
                            word_list.append(word)
                            continue
                        twords.append(word_list)
                    
                # 토픽 단어 가져오기, 토픽을 csv로 저장하기
                data = {}
                for i, list in enumerate(twords):
                    data["Topic " + str(i)] = list
                df = pd.DataFrame(data)
                path_topic = path / company_id[j] /  "topic.csv"
                df.to_csv(path_topic, index=False, encoding="utf-8")
                
                # 날짜 형식 바꾸기
                da = pd.to_datetime(str(date), format='%Y%m%d')
                da = da.strftime('%Y-%m-%d')

                # 백터 값 가져오기
                theta = []
                with open(path / company_id[j] / "vector.txt.theta", "r", encoding='utf_8', errors='ignore') as fin:
                    for line in fin.readlines():
                        topic_vec = []
                        toks = line.split(" ")  # \n 지우기
                        del toks[-1]
                        #print(toks)

                        for vec in toks:
                            # try:
                            vec = float(vec)
                            # except ValueError:
                            #     continue
                            topic_vec.append(vec)


                #print(topic_vec)

                # match 시키기
                for data in sentiment_df.values:
                    aspect = data[0]
                    sentiment = data[1]
                    sentiment_date = data[2]
                    if sentiment_date == da:
                        # print(aspect , sentiment)

                        for i, topic_num in enumerate(df):
                            #print(df[k].values)
                            for word in df[topic_num].values:
                                #print(aspect)
                                if word == aspect:
                                    #print(aspect, sentiment, i)
                                    if sentiment == "Positive":
                                        topic_vec[i] = topic_vec[i] + 0.1
                                    elif sentiment == "Negative":
                                        topic_vec[i] = topic_vec[i] - 0.1


                theta.append(topic_vec)
                theta = np.array(theta)
                #print("theta", theta)
                #print(j)
                #print(topic_vector_data)
                
                topic_vector_data[j] = theta
        tmp = np.concatenate(topic_vector_data).reshape(
            1, int(topics) * len(company_id)
        )
        topic_vector.append(tmp)
        #print(topic_vector)

    #print("topic_vector", topic_vector)
    topic_vector = np.concatenate(topic_vector)
    #print("topic_vector", topic_vector)
    output(topic_vector, path_pkl + args.output)


if __name__ == "__main__":

    args = parser_args()
    path_news = pathlib.Path(args.news)                     # data\output_morphological.csv
    path_stock = pathlib.Path(path_data + args.stock)       # data\price
    path_vector = pathlib.Path(args.vector)                 # data\vector
    path_list = pathlib.Path(path_data + args.company_list)  # data\stock_index.txt
    path_sentiment = "./../../data/sentiment/sentiment.csv"

    # 가격 데이터 이름
    # company_id = ['Dow', 'Nasdaq', 'SnP']
    company_id = read_company_id(path_stock)

    # 'S&P500' 'NASDAQ', 'Dow 30'
    company_list = read_company_list(path_list)

    # news = 데이터 마이닝 후의 텍스트 데이터
    # date_list = 가격 데이터의 날짜
    news, date_list = read_csv(path_news, path_stock)

    # stock_index 리스트에 있는 단어가 포함된 뉴스만 뽑아오기
    company_index_dict = extract_news(company_list, date_list, news)

    # 폴더 만들기 (data\vector, 가격 데이터의 날짜,
    # stock_index의 단어가 포함된 뉴스, ['Dow', 'Nasdaq', 'SnP'], 데이터 마이닝 후의 텍스트 데이터)
    make_folder(path_vector, date_list, company_index_dict, company_id, news)

    # LDA 실행
    LDA(args, company_id, date_list, company_index_dict)
    
    # ▼▼▼▼ABSA와 합치는 작업▼▼▼▼
    sentiment_df = sentiment_analysis(path_sentiment)

    # vector 값 바꾸는 함수
    replace_topic(args, date_list, company_index_dict, company_id, sentiment_df)

