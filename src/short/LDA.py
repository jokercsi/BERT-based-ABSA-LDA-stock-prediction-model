import os
import subprocess as sub
import sys


def LDA_est(
    path_lda, alpha, beta, ntopics, niters, savestep, twords, dfile, path_return
):
    os.chdir(path_lda)

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
        print("外部プログラムの実行に失敗しました", file=sys.stderr)

    os.chdir(path_return)


def LDA_inf(path_lda, dirc, model, niters, twords, dfile, path_return):

    os.chdir(path_lda)

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
    file = " -dfile " + str(dfile)

    try:
        sub.run(cmd + file, shell=True, check=True)

    except sub.CalledProcessError:
        print("外部プログラムの実行に失敗しました", file=sys.stderr)

    os.chdir(path_return)
