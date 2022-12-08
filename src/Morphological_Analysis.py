import argparse
import pathlib # pathlib 모듈의 기본 아이디어는 파일시스템 경로를 단순한 문자열이 아니라 객체로 다루자는 것입니다. 
import pandas as pd
import os
from tqdm import tqdm

# 형태소 분석 nltk
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')  #토큰화
nltk.download('wordnet')
nltk.download('omw-1.4')

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
    for f in news_csv:  # csv 파일 하나씩 읽기
        tmp = pd.read_csv(f, encoding="utf8")   # 각각 파일의 읽은 내용
        news = pd.concat([news, tmp], axis=0)   # 하나로 합치기
    news['Date'] = pd.to_datetime(news['Date'], errors='raise') # 시간 형식 통일하기
    news = news.sort_values("Date")
    news = news.reset_index(drop=True)
    return news



# 
def Morphological(article, stopwords):

    headline= article[0] # 뉴스 기사의 첫번째 컬럼

    headline = nltk.word_tokenize(headline) # 토큰화

    mwtokenizer = nltk.MWETokenizer(separator='')
    mwtokenizer.add_mwe(('S', '&', 'P'))
    headline = mwtokenizer.tokenize(headline)

    headline = [word for word in headline if word not in (stopwords)]   #against, be, of, a, in, to 등의 단어가 제거 된걸 확인 할 수 있다.
    headline = [WordNetLemmatizer().lemmatize(word, pos='v') for word in headline]  #표제어 추출을 수행합니다. 표제어 추출로 3인칭 단수 표현을 1인칭으로 바꾸고, 과거 현재형 동사를 현재형으로 바꿉니다.
    #headline = [word for word in headline if len(word) > 2]     #길이가 2이하인 단어에 대해서 제거하는 작업을 수행합니다.

    text_result = ""    # list를 string 형태로 바꾸기 위함

    for word in headline:
        text_result += word + " "

    return text_result

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



    #print(news.values)
    articles = []

    # tqdm() 함수 : 진행상황을 보여준다
    for article in tqdm(news.values):
        #print(article)
        word_list = Morphological(article, stopwords)
        articles.append(word_list)
        #print(articles)


    for i, article in enumerate(articles):
        # print(i, article)
        news.loc[i, "after_headlines"] = article

    news.to_csv(path_output, encoding="utf-8")


# 함수들 호츌
if __name__ == "__main__":
    args = parser_args()
    get_wordlist_in_text(args)