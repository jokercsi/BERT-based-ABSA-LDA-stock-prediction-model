import os
import subprocess as sub  # 다른 프로세스를 실행하고 출력 결과를 가져올 수 있게 해주는 라이브러리
import sys


def LDA_est(
    path_lda, alpha, beta, ntopics, niters, savestep, twords, dfile, path_return
):
    os.chdir(path_lda)
    # print(path_lda) ./../../lda/src

    # LDA command
    cmd = (
        "lda -est"
        + " -alpha "
        + str(alpha)
        + " -beta "
        + str(beta)
        + " -ntopics "
        + str(ntopics)
        + " -niters "
        + str(niters)
        + " -savestep "
        + str(savestep)
        + " -twords "
        + str(twords)
    )
    file = " -dfile " + str(dfile)

    try:
        sub.run(cmd + file, shell=True, check=True)

    except sub.CalledProcessError:
        print("外部プログラムの実行に失敗しました 1", file=sys.stderr)

    os.chdir(path_return)


def LDA_inf(path_lda, dirc, model, niters, twords, dfile, path_return):

    os.chdir(path_lda)

    # cmd example: lda -inf -dir ./data/train/ -model model-final -niters 30 -twords 100
    cmd = (
        "lda -inf"
        + " -dir "
        + dirc
        + " -model "
        + model
        + " -niters "
        + str(niters)
        + " -twords "
        + str(twords)
    )
    # file example: -dfile ./../vector/20xx12xx/SnP/vector.txt
    file = " -dfile " + str(dfile)

    try:
        # cmd example + file example
        sub.run(cmd + file, shell=True, check=True)

    except sub.CalledProcessError:
        print("外部プログラムの実行に失敗しました 2", file=sys.stderr)

    os.chdir(path_return)
