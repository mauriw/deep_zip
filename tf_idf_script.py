import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf

import nltk
from nltk.tokenize import word_tokenize
from nltk.collocations import *

from string import punctuation

# import tensorflow_datasets as tfds
from collections import defaultdict
import pickle as pkl
import os
from tqdm import tqdm
from utils import get_data

filter = [punctuation] + ["n"]


N_DOC = 10000 # how many documents to process
GRAMS = 123 # 123 if you want to generate all [1-3]-grams, 1 if only 1-gram, etc.

"""
Cleans an input wikitext document (string) and returns the list of cleaned
tokens.
"""
def gen_cleaned_wiki(text):
    w = word_tokenize(text)
    first = True
    toks = []
    for k in w:
        if k in filter or len(k) < 2: continue
        add = [word for word in k.split("\\n") if word not in filter and len(word) > 1 and word[0] != "x"]
        for a in add:
            # print(a)
            a = a.split("\\")
            for word in a:
                if first and len(word):
                    word = word[2:]
                    first = False
                if len(word) > 1 and word[0] != "x": toks.append(word)
    return toks

"""
Generates a defaultdict produces the IDF scores for common n_grams using n_doc
wikipedia articles from the English wikipedia database. The resulting
defaultdict is stored in a pickle file.

Arguments:
    n_doc: int
    n_grams: and int 1-3 or combination (to generate all grams, pass in 123)
"""
def output_idf(n_doc=1000, n_grams=123):
    n_opt = str(n_grams)
    if str(2) in n_opt:
        bi_measures = nltk.collocations.BigramAssocMeasures()
    if str(3) in n_opt:
        tri_measures = nltk.collocations.TrigramAssocMeasures()

    DF = defaultdict(int)

    # ds = tfds.load('wikipedia/20201201.en', split='train', shuffle_files=True)
    ds = get_data()
    DS_SIZE = len(ds)

    pbar = tqdm(total=n_doc)
    for i, text in enumerate(ds):
        if n_doc and i == n_doc: break # stop at n_doc document
        # text = str(d["text"].numpy())
        toks = gen_cleaned_wiki(text)

        set_toks = []
        if str(1) in n_opt:
            set_toks += set(toks)

        if str(2) in n_opt:
            finder = BigramCollocationFinder.from_words(toks)
            set_toks += [t[0] for t in finder.score_ngrams(bi_measures.raw_freq)]

        if str(3) in n_opt:
            finder = TrigramCollocationFinder.from_words(toks)
            set_toks += [t[0] for t in finder.score_ngrams(tri_measures.raw_freq)]


        for word in set_toks:
            DF[word] += 1
        pbar.update(1)

    pbar.close()

    # calc IDFs
    IDF = defaultdict(int)
    N = n_doc if n_doc else len(ds)
    for word in DF:
        IDF[word] = np.log(N/DF[word]) + 1 # TODO: checkout if 1 is necessary

    # print(IDF)

    # save in file
    with open("inv_doc_freq_" + str(n_doc) + "_" + n_opt + ".p", "wb") as f:
        x = pkl.dump(IDF, f)
        f.close()

def tf_idf():
    output_idf(N_DOC, GRAMS)

if __name__ == '__main__':
    nltk.download('punkt')
    tf_idf()

# e.g. code to open produced file
# with open("doc_freq.p", "rb") as f:
#     d = pkl.load(f)

# add datasets to the requirements.txt
# pip install -q tfds-nightly tensorflow matplotlib
