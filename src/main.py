import decompressor
import dataloader
import numpy as np

# from transformers import BertTokenizer, BertForMaskedLM
from .mask import write_all

COMPRESSION_FACTOR = 0.1

def accuracy(txt, decompressed_txt, mask_indices):
    if len(mask_indices) == 0:
        return 1.0
    split_txt = txt.split()
    split_decompressed_txt = decompressed_txt.split()
    num_right = 0
    for i in mask_indices:
        if split_txt[i] == split_decompressed_txt[i]:
            num_right += 1
    return num_right / len(mask_indices)

def compression(txt, compressed_txt):
    return len(compressed_txt) / len(txt)

if __name__ == '__main__':
    model = BertForMaskedLM.from_pretrained('bert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    x = 0
    for txt in dataloader.get_data():
        if not txt:
            continue
        print(txt)
        print('-' * 100)
        import random
        split_txt = txt.split()
        n = len(split_txt)
        mask_indices = np.sort(np.random.choice(n, size=int(n * COMPRESSION_FACTOR), replace=False))
        compressed_txt = ' '.join(['[MASK]' if i in mask_indices else w for i, w in enumerate(split_txt)])
        print(compressed_txt)
        print('-' * 100)
        decompressed_txt = decompressor.decompress(compressed_txt, model, tokenizer)
        print(decompressed_txt)
        print('-' * 100)
        print(f"Accuracy: {accuracy(txt, decompressed_txt, mask_indices)}")
        print('-' * 100)
        print(f"Compression: {compression(txt, compressed_txt)}")
        print('-' * 100)
        x += 1
        if x == 2:
            break
