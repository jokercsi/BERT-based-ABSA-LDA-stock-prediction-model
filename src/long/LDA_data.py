import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="./../../data/Morphological_data.csv")
    parser.add_argument("-train", "--train", default="./../../data/train/train.txt")
    parser.add_argument("-test", "--test", default="./../../data/test/test.txt")

    return parser.parse_args()


def make_data(data, path):
    file_out = open(path, "w", encoding="utf_8")
    file_out.write(str(len(data)))
    file_out.write("\n")
    for i in data.index:
        file_out.write(data["本文"][i])
        file_out.write("\n")
    file_out.close()


if __name__ == "__main__":
    args = parser_args()
    news = pd.read_csv(args.input, encoding="utf8")
    train, test = train_test_split(news, shuffle=False, test_size=None, train_size=0.7)
    make_data(train, args.train)
    # make_data(test, args.test)
