import constants
from collections import Counter

def compress(txt, window_size=10):
    masked = []
    split_txt = txt.split()
    for i in range(0, len(split_txt), window_size):
        window = split_txt[i:i + window_size]
        j, _ = max(enumerate(window), key=lambda w: len(w[1]))
        window[j] = constants.MASK
        masked += window
    return ' '.join(masked)

def global_compress(txt, num_words=100):
    split_txt = txt.split()
    word2count = Counter(split_txt)
    total_chars = [(word, count * len(word)) for word, count in word2count.items()]
    top_compressions = sorted(total_chars, key=lambda x: x[1], reverse=True)[:num_words]
    compression_words = {word for word, _ in top_compressions}
    masked = [word if word not in compression_words else constants.MASK for word in split_txt ]
    return ' '.join(masked), list(compression_words)
    