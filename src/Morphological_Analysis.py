import argparse
import pathlib # pathlib 모듈의 기본 아이디어는 파일시스템 경로를 단순한 문자열이 아니라 객체로 다루자는 것입니다. 
import pandas as pd
import os

# 형태소 분석 nltk
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag

def parser_args():

    #현재 위치 확인
    cwd = os.getcwd()
    print(cwd)

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--stopword", default="./data/stopword.txt")               # Stopword 지정
    parser.add_argument("-n", "--news", default="./data/text")                           # Dateset 지정
    parser.add_argument("-o", "--output", default="./data/output_morphological.csv")     # OUTPUT 지정

    return parser.parse_args()

def stopword(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        stopwords = [w.strip() for w in f]
        stopwords = set(stopwords)
        f.close()
    return stopwords

# glob() 함수 : 파라미터에 명시된 저장 경로와 패턴에 해당하는 파일명을 리스트 형식으로 반환한다.
def make_df(data_path):

    news_csv = data_path.glob("*.csv")      #csv 파일들의 이름만 news_csv에 리스트에 저장된다.
    # pd.set_option("display.max_colwidth", 5400)
    # pd.set_option("display.max_rows", 5000)
    news = pd.DataFrame()
    for f in news_csv:
        print(f)
        tmp = pd.read_csv(f, encoding="utf8")   # 각각 파일의 읽은 내용
        news = pd.concat([news, tmp], axis=0)   # 하나로 합치기
    news['Date'] = pd.to_datetime(news['Date'], errors='raise') # 시간 형식 통일하기
    news = news.sort_values("Date")
    news = news.reset_index(drop=True)
    return news



# 
def Morphological(article, stopwords):

    mecab = MeCab.Tagger(r'-Owakati -d "./../data/mecab-ipadic-neologd"')
    text = article[8]
    text_out = ""
    text_result = ""
    node = mecab.parseToNode(text)
    while node:
        f = node.feature.split(",")
        for stopword in stopwords:
            if f[6] == stopword:
                f[6] = "*"
        if (f[6] != "*") and (f[0] == "名詞"):
            text_out += f[6] + " "
        node = node.next
    text_out = text_out.replace("\u3000", "").replace("\n", "")
    text_out = re.sub(
        r"[0123456789０１２３４５６７８９！＠＃＄％＾＆\-|\\＊\“（）＿■×※⇒—●(：〜＋=)／*&^%$#@!~`){}…\[\]\"\'\”:;<>?＜＞？、。・,./『』【】「」→←○]+",
        "",
        text_out,
    )
    sentense = text_out.split(" ")
    sentense = list(filter(("").__ne__, sentense))

    for word in sentense:
        text_result += word + " "

    return text_result, len(sentense)


def get_wordlist_in_text(args):

    # path 지정
    path_stopwords = pathlib.Path(args.stopword)
    path_data = pathlib.Path(args.news)
    path_output = pathlib.Path(args.output)

    # print(path_stopwords)
    # print(path_data)
    # print(path_output)

    # 함수 사용
    news = make_df(path_data)   # 뉴스데이터 통합하고 가져오기
    #news.to_csv('output.csv',index=False, encoding="utf-8")
    stopwords = stopword(path_stopwords)    # stopword 가져오기
    #print(news)
    #print(stopwords)


    articles = []
    word_len = []

    for sentense in tqdm(news.values):
        word_list = Morphological(sentense, stopwords)
        word_len.append(word_list[1])
        articles.append(word_list[0])

    # for i, (article, word) in enumerate(zip(articles, word_len)):
    #     news.loc[i, "本文"] = article
    #     news.loc[i, "文字数"] = word

    # news.to_csv(path_output)


# 함수들 호츌
if __name__ == "__main__":
    args = parser_args()
    get_wordlist_in_text(args)