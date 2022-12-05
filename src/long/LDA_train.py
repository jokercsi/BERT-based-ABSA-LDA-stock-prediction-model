import argparse

from LDA import LDA_est


def parser_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-lda", "--lda", default="./../../lda/src", help="lda/srcまでの相対パス"
    )
    parser.add_argument(
        "-train",
        "--train",
        default="./../../data/train/train.txt",
        help="lda/srcから推定データまでの相対パス",
    )
    parser.add_argument(
        "-current",
        "--current",
        default="./../../src/long/",
        help="lda/srcから現在のフォルダまでの相対パス",
    )
    parser.add_argument("-alpha", "--alpha", default=0.1)
    parser.add_argument("-beta", "--beta", default=0.1)
    parser.add_argument("-ntopics", "--topics", default=100)
    parser.add_argument("-niters", "--niters", default=1000)
    parser.add_argument("-savesteps", "--savesteps", default=500)
    parser.add_argument("-twords", "--twords", default=100)

    return parser.parse_args()


if __name__ == "__main__":
    args = parser_args()
    LDA_est(
        args.lda,
        args.alpha,
        args.beta,
        args.topics,
        args.niters,
        args.savesteps,
        args.twords,
        args.train,
        args.current,
    )
