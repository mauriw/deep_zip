import compressor
import constants
import copy
import data
import decompressor
import models
import numpy as np
import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import BertModel, BertTokenizer

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

def run_experiment(model, tokenizer):
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

base_training_args = {
    'epochs': 3,
    'batch_size': 32,
    'shuffle': True,
    'optimizer': torch.optim.Adam,
    'loss_fn': F.cross_entropy,
    'lr': 1e-3
}

if __name__ == '__main__':
    for directory in ['best_models', 'outputs', 'plots', 'results']:
        if not os.path.exists(directory):
            os.mkdir(directory)

    encoder_name = 'bert-base-cased'
    for compression_name in ['poesia', 'tf-idf']:
        for output_vocab_size in [300, 1000, 2000]:
            encoder = BertModel.from_pretrained(encoder_name)
            # TODO check if you can change tokenizer mask token
            tokenizer = BertTokenizer.from_pretrained(encoder_name)
            if compression_name == 'poesia':
                compression_tokens = compressor.preprocess_poesia(data.get_dataset('train'), output_vocab_size)
            elif compression_name == 'tf-idf':
                compression_tokens = compressor.preprocess_tf_idf(data.get_dataset('train'), output_vocab_size)
            print("Compression tokens:")
            print(compression_tokens)
            print('*' * 100)

            tokenizer.add_tokens(compression_tokens)
            compression_ids = torch.tensor(tokenizer.convert_tokens_to_ids(compression_tokens)).to(constants.DEVICE)
            compression_id_to_idx = {t.item(): i for i, t in enumerate(compression_ids)}
            model = models.BertFinetune(encoder, constants.PRETRAINED_BERT_OUTPUT_HIDDEN_SIZE, len(compression_tokens))

            train_dset = data.CompressedDataset(data.get_dataset('train'), tokenizer, compression_ids, compression_id_to_idx)
            val_dset = data.CompressedDataset(data.get_dataset('val'), tokenizer, compression_ids, compression_id_to_idx)
            test_dset = data.CompressedDataset(data.get_dataset('test'), tokenizer, compression_ids, compression_id_to_idx)
            train_dataloader = torch.utils.data.DataLoader(train_dset, shuffle=base_training_args['shuffle'], batch_size=base_training_args['batch_size'])
            val_dataloader = torch.utils.data.DataLoader(val_dset, shuffle=base_training_args['shuffle'], batch_size=base_training_args['batch_size'])
            test_dataloader = torch.utils.data.DataLoader(test_dset, shuffle=False, batch_size=base_training_args['batch_size'])

            models.train(model, tokenizer, train_dataloader, val_dataloader, compression_ids,
                        f'{encoder_name}_{compression_name}_{output_vocab_size}', base_training_args)
            print('=' * 100)
            models.eval(model, tokenizer, test_dataloader, compression_ids,
                        f'{encoder_name}_{compression_name}_{output_vocab_size}', base_training_args['loss_fn'])
            print('*' * 100)
