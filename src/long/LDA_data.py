import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def parser_args():

    #현재 위치 확인
    cwd = os.getcwd()
    print("current location is :", cwd)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="./data/output_morphological.csv")   # input data PATH
    parser.add_argument("-train", "--train", default="./data/train/train.txt")        # train data PATH
    parser.add_argument("-test", "--test", default="./data/test/test.txt")            # test data PATH

    return parser.parse_args()


# txt 파일로 만드는 함수
def make_data(data, path):
    file_out = open(path, "w", encoding="utf_8")
    file_out.write(str(len(data)))
    file_out.write("\n")
    for i in data.index:
        file_out.write(data["after_headlines"][i])
        file_out.write("\n")
    file_out.close()


if __name__ == "__main__":
    args = parser_args()
    news = pd.read_csv(args.input, encoding="utf8") # input 파일 읽기

    train, test = train_test_split(news, shuffle=False, test_size=None, train_size=0.7) # train 과 test 나누기
    make_data(train, args.train)    # train 파일 작성
    make_data(test, args.test)      # test 파일 작성
