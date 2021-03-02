from pathlib import Path
from typing import List

MASK = '[MASK]'
# WIKI_DIR = Path.cwd() / Path('data/wikitext-2/')
# MASKED_DIR = Path.cwd() / Path('data/masked/')

# def get_wiki(split):
#     assert split == 'train' or split == 'valid' or split == 'test'
#     path = WIKI_DIR / (split + '.tokens')
#     return path.read_text()

# def get_masked(split):
#     assert split == 'train' or split == 'valid' or split == 'test'
#     path = MASKED_DIR / (split + '.tokens')
#     return path.read_text()

def compress(txt, window_size=10):
    masked = []
    split_txt = txt.split()
    for i in range(0, len(split_txt), window_size):
        window = split_txt[i:i + window_size]
        j, _ = max(enumerate(window), key=lambda w: len(w[1]))
        window[j] = MASK
        masked += window
    return ' '.join(masked)

# def write_split(split, window_size=10):
#     text = sliding_window(split, window_size)
#     path = MASKED_DIR / (split + '.tokens')
#     path.write_text(text)

# def write_all(window_size=10):
#     write_split('train', window_size)
#     write_split('valid', window_size)
#     write_split('test', window_size)

if __name__ == '__main__':
    write_all()
