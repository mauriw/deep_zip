import datasets
from collections import Counter
import constants

"""
Returns raw wikitext train input
"""
def get_data():
    # return wikitext dataset
    dataset = datasets.load_dataset('wikitext', 'wikitext-2-v1')
    return dataset['train']['text']

def unique_words(ds):
    uniq = {w for txt in ds for w in txt.split()}
    print("Unique words:", len(uniq))
    return sorted(uniq)

# original global compress
def global_compress(txt, num_words=100):
    # returns:
    #   - masked txt (string)
    #   - list of compression words in alphabetical order (list of strings)
    split_txt = txt.split()
    word2count = Counter(split_txt)
    total_chars = [(word, count * len(word)) for word, count in word2count.items()]
    top_compressions = sorted(total_chars, key=lambda x: x[1], reverse=True)[:num_words]
    compression_words = {word for word, _ in top_compressions}
    masked = [word if word not in compression_words else constants.MASK for word in split_txt ]
    return ' '.join(masked), sorted(compression_words)
