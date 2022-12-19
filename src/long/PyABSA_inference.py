from pyabsa import ATEPCCheckpointManager
import argparse
import os
import pandas as pd
from nltk import sent_tokenize


def parser_args():

    # 현재 위치 확인
    cwd = os.getcwd()
    print("current location is :", cwd)

    parser = argparse.ArgumentParser()
    # input data PATH
    parser.add_argument("-i", "--input", default="./../../data/output_morphological.csv")

    return parser.parse_args()


def preprocessing_for_pyABSA(news):

    headlines = news[0]

    tokenized_text = sent_tokenize(headlines)
    print(tokenized_text)

    return tokenized_text


def pyABSA(news):
    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
        checkpoint='english',
        auto_device=True  # False means load model on CPU
    )
    df = news['after_headlines'].head()
    df_list = df.values.tolist()  # pandas to list

    # You can inference from a list of setences or a DatasetItem from PyABSA
    examples = df_list
    inference_source = examples
    atepc_result = aspect_extractor.extract_aspect(
        inference_source=inference_source,
        pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
    )

    return print(atepc_result)


# 함수들 호츌
if __name__ == "__main__":
    args = parser_args()
    news = pd.read_csv(args.input, encoding="utf-8")  # input 파일 읽기
    preprocessing_for_pyABSA(news)
    #pyABSA(news)
