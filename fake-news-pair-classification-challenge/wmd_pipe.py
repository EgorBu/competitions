from collections import Counter
import os


import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm_notebook as tqdm
from wmd import WMD

def main(args):
    # location of input-output
    data_dir = args.input
    train_loc = os.path.join(data_dir, "train.csv")
    test_loc = os.path.join(data_dir, "test.csv")
    train = pd.read_csv(train_loc)
    test = pd.read_csv(test_loc)
    nlp = spacy.load("en_core_web_lg")

    def extract_bow(data, text_col, id_col, uniq_tokens=None):
        documents = {}
        sent = {}
        if uniq_tokens is None:
            uniq_tokens = {}
        for i, line in tqdm(data.iterrows(), total=data.shape[0]):
            # TODO: remove after debugging
            sent[line[id_col]] = line[text_col]
            if i == 1000:
                # TODO: remove after experiments
                break

            text = nlp(line[text_col])
            tokens = [t for t in text if t.is_alpha and not t.is_stop]
            orths = {t.text: t.orth for t in tokens}
            words = Counter(t.text for t in tokens if t.text in nlp.vocab)
            sorted_words = sorted(words)
            documents[line[id_col]] = (
                line[id_col], [orths[t] for t in sorted_words],
                np.array([words[t] for t in sorted_words], dtype=np.float32)
            )
        return documents, uniq_tokens, sent

    tid1_nlp, uniq_tokens, tid1_sent = extract_bow(train, text_col="title1_en", id_col="tid1")
    tid2_nlp, uniq_tokens, tid2_sent = extract_bow(train, text_col="title2_en", id_col="tid2",
                                                   uniq_tokens=uniq_tokens)

    class SpacyEmbeddings(object):
        def __getitem__(self, item):
            return nlp.vocab[item].vector

    from wmd import TailVocabularyOptimizer

    tid1_calc = WMD(SpacyEmbeddings(), tid1_nlp, vocabulary_min=10,
                    vocabulary_optimizer=TailVocabularyOptimizer(1.))
    tid2_calc = WMD(SpacyEmbeddings(), tid2_nlp, vocabulary_min=3)

