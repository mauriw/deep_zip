from pathlib import Path
from typing import List

MASK = '[MASK]'
WIKI_DIR = Path.cwd() / Path('data/wikitext-2/')
MASKED_DIR = Path.cwd() / Path('data/masked/')

def get_wiki(split):
    assert split == 'train' or split == 'valid' or split == 'test'
    path = WIKI_DIR / (split + '.tokens')
    return path.read_text()

def get_masked(split):
    assert split == 'train' or split == 'valid' or split == 'test'
    path = MASKED_DIR / (split + '.tokens')
    return path.read_text()

def mask(window: list):
    i, word = max(enumerate(window), key=lambda x: len(x[1]))
    window[i] = MASK
    print(word, 'has been replaced by a mask')
    return ' '.join(window)

def sliding_window(split, window_size=10):
    masked = []
    corpus = get_wiki(split)
    c = corpus.split()
    for i in range(0, len(c), window_size):
        start = i
        end = i + 10 
        if end > len(c): 
            break
        window = c[start:end]
        masked.append(mask(window))
    return ' '.join(masked)

def write_split(split, window_size=10):
    text = sliding_window(split, window_size)
    path = MASKED_DIR / (split + '.tokens')
    path.write_text(text)

def write_all(window_size=10):
    write_split('train', window_size)
    write_split('valid', window_size)
    write_split('test', window_size)

if __name__ == '__main__':
    write_all()
