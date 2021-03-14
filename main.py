import compressor
import constants
import decompressor
import dataloader
import numpy as np

from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM

# Return num_correct_masks, num_total_masks, num_correct, num_total

"""
Calculates the acc of text decompression (correctly un-masked), returns number
correctly masked, number of masked toks, the number of correct tokens (in total)
and the doc len.

Arguments:
    txt: input text string
    compressed_text: input compressed text string
    decompressed_text: input decompressed string
"""
def accuracy(txt, compressed_txt, decompressed_txt):
    mask_indices = [i for i, w in enumerate(compressed_txt.split()) if w == constants.MASK]
    split_txt = txt.split()
    split_decompressed_txt = decompressed_txt.split()
    num_right_masked = sum(split_txt[i] == split_decompressed_txt[i] for i in mask_indices)
    return num_right_masked, len(mask_indices), len(split_txt) - len(mask_indices) + num_right_masked, len(split_txt)


if __name__ == '__main__':

    # get model and tokenizer
    model = BertForMaskedLM.from_pretrained('bert-base-cased')
    model.to(constants.DEVICE)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


    # calculate system perfomrance (compression & decompression)
    num_correctly_decompressed = 0
    num_compressed = 0
    num_correct_overall = 0
    num_overall = 0
    total_compressed_len = 0
    total_original_len = 0

    # iterate over dataset
    for txt in tqdm(dataloader.get_data()):

        if len(txt.strip()) == 0: # skip empty text
            continue

        txt = txt[:constants.PRETRAINED_BERT_MAX_LEN]
        compressed_txt = compressor.compress(txt) # get compression

        # decompress compression
        decompressed_txt = decompressor.decompress(compressed_txt, model, tokenizer)

        # calculate acc
        accuracy_stats = accuracy(txt, compressed_txt, decompressed_txt)
        num_correctly_decompressed += accuracy_stats[0]
        num_compressed += accuracy_stats[1]
        num_correct_overall += accuracy_stats[2]
        num_overall += accuracy_stats[3]
        total_compressed_len += len(compressed_txt.encode('utf8'))
        total_original_len += len(txt.encode('utf8'))

    # report run stats
    print("-" * 150)
    print(f"Mask Accuracy: {num_correctly_decompressed / num_compressed}")
    print(f"Total Accuracy: {num_correct_overall / num_overall}")
    print(f"Total Compression: {total_compressed_len / total_original_len}")
    print(f"Corpus size: {total_original_len} characters, {num_overall} words")
