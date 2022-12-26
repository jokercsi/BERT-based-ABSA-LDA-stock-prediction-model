import argparse
import os
from itertools import chain

import pandas as pd
from nltk import sent_tokenize
from    nltk import word_tokenize
from pyabsa import ATEPCCheckpointManager
from tqdm import tqdm

path_sentiment = "./../../data/sentiment/sentiment.csv"
path_stopwords = "./../../data/stopword.txt"

def parser_args():

    parser = argparse.ArgumentParser()
    # Stopword 지정
    parser.add_argument("-s", "--stopword", default="./../data/stopword.txt")
    # input data PATH
    parser.add_argument("-i", "--input", default="./../../data/output_morphological.csv")

    return parser.parse_args()


def preprocessing_for_pyABSA(news):

    headlines = news['Headlines']
    date = news['Date']
    # print(type(headlines.values))
    # print(date.values)
    
    date_list = date.values.tolist()
    articles = []
    for article in tqdm(headlines.values):
        line_list = Morphological(article)
        articles.append(line_list)
        # print(articles)

    # 날짜랑 기사 갯수 맞추기
    for i, v in enumerate(articles):
        #print(len(v))
        count = len(v)
        if count > 1:
            #print(date_list[i])
            for j in range(count-1):
                date_list.insert(i, date_list[i])
            
    return articles, date_list


def stopword(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        stopwords = [w.strip() for w in f]
        stopwords = set(stopwords)
        f.close()
    return stopwords


def Morphological(headline):
    stopwords = stopword(path_stopwords)

    tokenized_as_line = sent_tokenize(headline)
    #print(len(tokenized_as_line))

    rm_stopwords = []
    for line in tokenized_as_line:
        rm_stopwords.append(' '.join(w for w in word_tokenize(line) if w.lower() not in stopwords))
    
    #print(rm_stopwords)
    return rm_stopwords


def pyABSA(articles_list):
    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
        checkpoint='english',
        auto_device=True  # False means load model on CPU
    )

    # You can inference from a list of setences or a DatasetItem from PyABSA
    inference_source = articles_list
    atepc_result = aspect_extractor.extract_aspect(
        inference_source=inference_source,
        pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
    )

    return atepc_result


def ABSA(articles_list, date_list):

    aspect = []
    sentiment = []
    date = []
    result = pyABSA(articles_list)
    for idx, i in enumerate(result):
        if i['aspect'] != []:
            if len(i['sentiment'])>1:
                for j, v in zip(i['aspect'], i['sentiment']):
                    #print(j , v)
                    aspect.append(j)
                    sentiment.append(v)
                    date.append(date_list[idx])
            else:
                #print(i['aspect'][0], i['sentiment'][0])
                aspect.append(i['aspect'][0])
                sentiment.append(i['sentiment'][0])
                date.append(date_list[idx])
    #print(len(date), len(sentiment), len(aspect))
    data = {'Aspect':  aspect,
            'Sentiment': sentiment,
            'Date': date,
            }
    df = pd.DataFrame(data)
    df.to_csv(path_sentiment, index=False, encoding="utf-8")
    #print(df)


# 함수들 호츌
if __name__ == "__main__":
    args = parser_args()
    
    news = pd.read_csv(args.input, encoding="utf-8")  # input 파일 읽기
    articles_list, date_list = preprocessing_for_pyABSA(news)

    #기사 문장이 여러개로 나눠지면 날짜도 플러스 하는 메소드 만들기

    articles_list = list(chain(*articles_list))
    #print(len(articles_list), len(date_list))

    ABSA(articles_list,date_list)
