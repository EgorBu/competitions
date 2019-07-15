import argparse
import os
from multiprocessing import Pool

import pandas as pd
import spacy
from tqdm import tqdm
import wmd

nlp_ = None  # lazy loading when it's needed in pipeline


def calculate_nlp_features(doc1, doc2):
    """
    Calculate different statistics about doc1 & doc2.

    :param doc1: column "title1_en".
    :param doc2: column "title2_en".
    :return: list of features.
    """
    stats = []

    # number of tokens
    stats.append(len(doc1))
    stats.append(len(doc2))

    # number of common tokens
    stats.append(len(set(doc1).intersection(doc2)))

    # number of common tokens lowercase
    stats.append(len(set(map(lambda x: str(x).lower(), doc1))
                     .intersection(map(lambda x: str(x).lower(), doc2))))

    return stats


def compute_similarity(line):
    """
    Calculate word mover's distance for pair of documents in line (English only).

    :param line: line from dataframe.
    :return: pair_id, WMD, features from calculate_nlp_features
    """
    line = line[1]  # skip first index
    pair_id = line["id"]
    doc1 = nlp_(line["title1_en"])
    doc2 = nlp_(line["title2_en"])
    try:
        wmd_res = doc1.similarity(doc2)
    except Exception as e:
        print(e)
        wmd_res = -1
    return (pair_id, wmd_res, *calculate_nlp_features(doc1, doc2))


def compute_similarity_dataframe(data, n_workers):
    with Pool(n_workers) as p:
        res = list(tqdm(p.imap(compute_similarity, data.iterrows()), total=len(data.shape)))

    columns = ["id", "wmd", "len_title1_en", "len_title2_en", "intersect_t1_t2",
               "intersect_t1_t2_lower"]

    return pd.DataFrame(data=res, columns=columns)


def pipeline(args):
    # load NLP and update globals
    global nlp_
    nlp_ = spacy.load(args.language)
    nlp_.add_pipe(wmd.WMD.SpacySimilarityHook(nlp_), last=True)
    # location of input-output
    data_dir = args.input
    train_loc = os.path.join(data_dir, "train.csv")
    test_loc = os.path.join(data_dir, "test.csv")
    train = pd.read_csv(train_loc)
    test = pd.read_csv(test_loc)
    # compute similarity and other features, save results
    # train
    train_res = compute_similarity_dataframe(train, n_workers=args.n_workers)
    train_res.to_csv(os.path.join(args.output, "train_sim_feat.csv"), index=False)
    # test
    test_res = compute_similarity_dataframe(test, n_workers=args.n_workers)
    test_res.to_csv(os.path.join(args.output, "test_sim_feat.csv"), index=False)


def create_parser() -> argparse.ArgumentParser:
    """
    Initialize the command line argument parser.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", default="input/",
                        help="Path to directory with input data.")
    parser.add_argument("-o", "--output", default="output/",
                        help="Directory where to store results.")
    parser.add_argument("-n", "--n-workers", default=2, type=int,
                        help="Number of workers to use.")
    parser.add_argument("-l", "--language", default="en_core_web_lg", type=str,
                        help="Default value to load spacy.")

    return parser


def main():
    """Entry point."""
    parser = create_parser()
    pipeline(parser.parse_args())


if __name__ == "__main__":
    main()
