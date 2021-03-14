import constants
import numpy as np
from collections import Counter, defaultdict
import pickle as pkl
import plotly.figure_factory as ff
import plotly.express as px
from tf_idf_script import output_idf
from utils import get_data
import os
from tqdm import tqdm

"""
Compress input txt string, window_size to select longest word and mask it.

Arguments:
    txt: input string text
    window_size: int
"""
def compress(txt, window_size=10):
    masked = []
    split_txt = txt.split()
    for i in range(0, len(split_txt), window_size):
        window = split_txt[i:i + window_size]
        j, _ = max(enumerate(window), key=lambda w: len(w[1]))
        window[j] = constants.MASK
        masked += window
    return ' '.join(masked)


"""
Return list of words that have the greatest frequency * length

Arguments:
    dataset: list of strings
    vocab_size: int, length of list
"""
def preprocess_poesia(dataset, vocab_size=100):
    split_txt = ' '.join(dataset).split()
    word2count = Counter(split_txt)
    total_chars = [(word, count * len(word)) for word, count in word2count.items() if len(word) > 1 and word != '[UNK]']
    top_compressions = sorted(total_chars, key=lambda x: x[1], reverse=True)[:vocab_size]
    return sorted({word for word, _ in top_compressions})


"""
pre-processing for TF-IDF
"""
def preprocess_tf_idf(dataset, vocab_size=10000, threshold=0.1, factor_length=True): # tune threshold

    # calc TF-IDF
    IDF = get_idf()
    TF = Counter()
    scores= {}
    for text in dataset:
        toks = text.split()
        TF = Counter(toks)
        for i, tok in enumerate(toks):
            score = float(1)/TF[tok] * float(1)/(IDF[tok]+0.01) * len(tok)
            if factor_length: score = score * len(tok)
            if tok in scores: score = min(score, scores[tok])
            scores[tok] = score

    tfidf_pairs = [(word, count) for word, count in scores.items() if  len(word) != 1]
    top_compressions = sorted(tfidf_pairs, key=lambda x: x[1], reverse=True)[-vocab_size:]# [:vocab_size] #[-vocab_size:]
    compression_words = {word for word, _ in top_compressions}
    return sorted(compression_words)

"""
Returns maps of IDF scores for every word in the wikitext dataset
(generated using tf_idf_script.py)
"""
def get_idf():
    file = "inv_doc_freq_10000_123.p"

    if not os.path.isfile(file):
        print("Generating TF-IDF weights...one moment please.")
        output_idf()

    with open(file, "rb") as f:
        return pkl.load(f)

"""
Runs analysis of impact of IDF only compress list

next steps: integrate TF+len robustly
"""
def test():
    ds = get_data()
    comp_toks = preprocess_tf_idf(ds,vocab_size=10000)
    # comp_toks = preprocess_poesia(ds,vocab_size=10000)

    zero = 0
    lots = 0
    dist = []
    lens = []
    pbar = tqdm(total=len(ds))
    for i,d in enumerate(ds):
        # if len(lens) == 20: break
        toks = d.split()
        if len(toks) == 0 or (toks[0] is "=" and toks[1] is "="):
            pbar.update(1)
            continue

        lens.append(len(toks))
        comp = [word if word not in comp_toks else constants.MASK for word in toks]
        n_masked = len([word for word in comp if word is constants.MASK])
        dist.append(n_masked/float(len(d)))
        if n_masked == 0: zero += 1
        if n_masked >= round(len(toks)*0.8): lots += 1
        if len(lens) == 6:
            print("Sample Output: \n")
            print(comp)
        pbar.update(1)


    print("AVG SAMPLE LEN: ", np.mean(lens))
    print("AVG COMP SIZE: ", np.mean(dist))
    print("MAX COMP SIZE: ", max(dist))
    print("MIN COMP SIZE: ", min(dist))
    print("Generated ", zero, " compressions with no mask tokens.")
    print("Generated ", lots, " compressions where 80% or more of the tokens were compressed")

    df = px.data.tips()
    fig = px.histogram(dist, nbins=100)
    fig.show()

    print("DS TYPE", type(ds))

def data_analysis():
    ds = get_data()
    dist = Counter()
    for d in ds:
        dist += Counter(d)



if __name__ == "__main__":
    test()
