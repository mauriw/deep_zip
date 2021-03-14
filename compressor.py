import constants
from collections import Counter, defaultdict
import pickle as pkl
import plotly.figure_factory as ff
import plotly.express as px
from tf_idf_script import output_idf
from utils import get_data
import os

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
pre-processing for TF-IDF
"""
def preprocess_tf_idf(text=None, threshold=10000, factor_length=True): # tune threshold

    # utilize TF (not doing so currently)
    # toks = text.split() # consider using BERT tokenizer (and just not tokenize punct)
    # TF = Counter(toks)
    # scores = {}
    # for i, tok in enumerate(toks):
    #     score = TF[tok] * (IDF[tok] + 0.1)
    #     if factor_length: score = score * len(tok)
    #     # if score > threshold: # idf
    #         # toks[i] = constants.MASK
    #     scores[tok] = score

    # utilize IDF
    IDF = get_idf()
    idf_pairs = [(word, count) for word, count in IDF.items() if type(word) is not tuple and  len(word) != 1]
    top_compressions = sorted(idf_pairs, key=lambda x: x[1], reverse=True)[:threshold]
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
    comp_toks = preprocess_tf_idf(ds[5])
    masked = []
    dist = []
    for i,d in enumerate(ds):
        toks = d.split()
        comp = [word if word not in comp_toks else constants.MASK for word in toks]
        masked.append(comp)
        dist.append(len([word for word in comp if word is constants.MASK]))
        if i == 10: break

    df = px.data.tips()

    fig = px.histogram(dist)
    fig.show()


if __name__ == "__main__":
    test()
