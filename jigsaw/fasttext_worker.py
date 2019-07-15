import re
from uuid import uuid4

import numpy as np
from pyfasttext import FastText
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

np.random.seed(42)  # reproducibility

LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def modify_str_col(df, input_col, output_col):
    """
    Lower case, remove digits and punctuation/special chars except "'" & "-", strip,
    several spaces -> one space.
    :param df: dataframe
    :param input_col: name of column to use
    :param output_col: name of output column
    :return:
    """
    pattern = re.compile("[^\w_'-]+", re.UNICODE)
    df[output_col] = df[input_col]
    df[output_col] = (df[output_col].str.replace(pattern, " ").str.lower().str.strip()
                      .str.replace(re.compile(' {2,}'), ' '))
    df[output_col][df[output_col].str.len() == 0] = " "
    return df


def transform_fasttext_proba(probas):
    res = []
    for proba in probas:
        assert len(proba) == 2, proba
        if "NONE" in proba[1]:
            res.append(proba[0][1])
        else:
            res.append(proba[1][1])

    return res


def predict_fasttext_proba(model, df, k=2):
    values = list(map(lambda x: x + "\n", df["modified_comment_text"].tolist()))
    probas = model.predict_proba(values, k=k)
    return transform_fasttext_proba(probas)


def filter_kwargs(kwargs, prefix=""):
    if isinstance(kwargs, dict):
        attributes = kwargs.keys()
        getter = lambda x: kwargs[x]
    else:
        attributes = dir(kwargs)
        getter = lambda x: getattr(kwargs, x)

    attributes = [attr for attr in attributes if attr.startswith(prefix)]

    filtered_kwargs = {}
    for attr in attributes:
        filtered_kwargs[attr[len(prefix):]] = getter(attr)

    return filtered_kwargs


def train_pipeline(train, val, to_predict, args, proba_postfix):
    train_proba = {}
    val_proba = {}

    # train binary classifier for each class
    roc_aucs = []
    for col in LABEL_COLUMNS:
        # save text file for fasttext
        cols = ["modified_comment_text", "fasttext_label_" + col, "fasttext_label_NONE"]
        train[cols].to_csv(args["supervised_input"], sep=" ", index=False, quotechar=" ")

        # TODO: make kwargs for each label
        fasttext_kwargs = filter_kwargs(args, prefix="supervised_")
        classifier = FastText()
        classifier.supervised(**fasttext_kwargs)
        train_proba[col] = predict_fasttext_proba(classifier, train, k=2)

        val_proba[col] = predict_fasttext_proba(classifier, val, k=2)

        for data in to_predict:
            data[col + "_proba_" + proba_postfix] = predict_fasttext_proba(classifier, data, k=2)

        train_score = roc_auc_score(train[col].tolist(), train_proba[col])
        score = roc_auc_score(val[col].tolist(), val_proba[col])
        roc_aucs.append(score)
        print(col, "roc_auc: train %s / val %s" % (train_score, score))
        print(col, "is done...")

    mean_roc_auc = np.mean(roc_aucs)
    print("mean roc_auc", mean_roc_auc)

    return mean_roc_auc


def pipeline(args):
    """
    args.train_path = "input/train.csv.zip"
    args.test_path = "input/test.csv.zip"
    args.n_splits = 2

    args.save_proba = "probas/"


    args.supervised_input = "output/train_fasttext.txt"

    args.supervised_output = "model"
    args.supervised_lr = 0.1
    args.supervised_lrUpdateRate = 5
    args.supervised_dim = 150
    args.supervised_ws = 5
    args.supervised_epoch = 15
    args.supervised_minCount = 5
    args.supervised_neg = 5
    args.supervised_wordNgrams = 2
    args.supervised_thread = 4
    args.supervised_label = "__label__"
    args.supervised_minn = 3
    args.supervised_maxn = 7
    args.supervised_pretrainedVectors = "input/crawl-300d-2M.vec"
    :param args:
    :return:
    """
    args = filter_kwargs(args)
    train = pd.read_csv(filepath_or_buffer=args["train_path"])
    test = pd.read_csv(filepath_or_buffer=args["test_path"])

    # create labels for fasttext
    for col in LABEL_COLUMNS:
        train["fasttext_label_" + col] = ""
        train["fasttext_label_" + col][train[col] == 1] += (args["supervised_label"] + col + " ")

    mask = train[LABEL_COLUMNS[0]] == 0
    for col in LABEL_COLUMNS[1:]:
        mask = mask & (train[col] == 0)
    train["fasttext_label_NONE"] = ""
    train["fasttext_label_NONE"][mask] += (args["supervised_label"] + "NONE" + " ")

    # simple comment preprocessing
    train = modify_str_col(train, "comment_text", "modified_comment_text")
    test = modify_str_col(test, "comment_text", "modified_comment_text")

    train_pred = train[["id", "modified_comment_text"]]
    test_pred = test[["id", "modified_comment_text"]]

    train_, val_ = train_test_split(train, test_size=0.1)

    roc_auc = train_pipeline(train_, val_, [train_pred, test_pred], args, str(0))

    uuid = str(uuid4())
    train_pred.to_pickle(args["save_proba"] + uuid + "_train.pkl")
    test_pred.to_pickle(args["save_proba"] + uuid + "_test.pkl")
    return roc_auc
