import gzip
import collections
import plotly.graph_objects as go

import constants
import data
import compressor

import torch

from transformers import BertTokenizer

"""
Generates a histogram of word counts
"""
def histogram(dataset, num_words):
    word2count = collections.Counter(' '.join(dataset).split())
    word_counts = [(word, count) for word, count in word2count.items()]
    sorted_counts = sorted(word_counts, key=lambda x: x[1], reverse=True)[:num_words]
    x, y = [t[0] for t in sorted_counts], [t[1] for t in sorted_counts]
    fig = go.Figure(data=go.Bar(x=x, y=y))
    fig.show()

"""
Prints compression scores
"""
def compression_scores():
    train_dataset = data.get_dataset('train')
    test_dataset = data.get_dataset('test')
    for compression_name in ['tf-idf', 'poesia']:
        for output_vocab_size in [300, 1000, 2000]:
            if compression_name == 'poesia':
                compression_tokens = compressor.preprocess_poesia(train_dataset, output_vocab_size)
            elif compression_name == 'tf-idf':
                compression_tokens = compressor.preprocess_tf_idf(train_dataset, output_vocab_size)
            
            original_len, compressed_len, gzip_len, gzip_compressed_len = 0, 0, 0, 0
            for sample in test_dataset:
                original_len += len(sample.encode('utf8'))
                gzip_len += len(gzip.compress(sample.encode('utf8')))

                compressed = [word if word not in compression_tokens else constants.NEW_MASK for word in sample.split()]
                compressed = ' '.join(compressed)

                compressed_len += len(compressed.encode('utf8'))
                gzip_compressed_len += len(gzip.compress(compressed.encode('utf8')))

            print('Compression name:', compression_name)
            print('Vocab size:', output_vocab_size)
            print()
            print('Compressed / Original:', compressed_len / original_len)
            print('Compressed & Gzip / Original:', gzip_compressed_len / original_len)
            print("*" * 100)
            print()
    print('Gzip / Original:', gzip_len / original_len)

def accuracy(base_fname):
    encoder_name = 'bert-base-cased'
    for compression_name in ['poesia', 'tf-idf']:
        for output_vocab_size in [300, 1000, 2000]:
            tokenizer = BertTokenizer.from_pretrained(encoder_name)
            if compression_name == 'poesia':
                compression_tokens = compressor.preprocess_poesia(data.get_dataset('train'), output_vocab_size)
                base_fname = f'outputs/{encoder_name}_{compression_name}_{output_vocab_size}_test'
            elif compression_name == 'tf-idf':
                compression_tokens = compressor.preprocess_tf_idf(data.get_dataset('train'), output_vocab_size)
                base_fname = f'outputs/{encoder_name}_{compression_name}_{output_vocab_size}_LARGER_test'
            tokenizer.add_tokens(compression_tokens)

            with open(f'{base_fname}_masked.txt') as f:
                masks = f.readlines()
            with open(f'{base_fname}_preds.txt') as f:
                preds = f.readlines()
            with open(f'{base_fname}_true.txt') as f:
                actuals = f.readlines()

            num_correct_masked = 0
            num_total_masked = 0
            num_correct = 0
            num_total = 0
            for i in range(len(preds)):
                masked, pred, actual = masks[i].replace('@', '[MASK]'), preds[i], actuals[i]
                if len(masked.strip()) == 0:
                    continue
                masked_indices = tokenizer(masked, return_tensors='pt')['input_ids'].squeeze() == tokenizer.mask_token_id
                pred_tokens = tokenizer(pred, return_tensors='pt', padding='max_length', max_length=masked_indices.numel())['input_ids'].squeeze()
                actual_tokens = tokenizer(actual, return_tensors='pt', padding='max_length', max_length=masked_indices.numel())['input_ids'].squeeze()

                if masked_indices.numel() < pred_tokens.numel():
                    masked_indices = torch.cat((masked_indices, torch.full((pred_tokens.numel() - masked_indices.numel(),), False)))

                num_correct_masked += torch.sum(pred_tokens[masked_indices] == actual_tokens[masked_indices]).item()
                num_total_masked += masked_indices.count_nonzero().item()
                num_correct += torch.sum(pred_tokens == actual_tokens).item()
                num_total += pred_tokens.numel()
            print(f"Compression name: {compression_name}")
            print(f"Vocab size: {output_vocab_size}")
            print()
            print(f"Masked accuracy: {num_correct_masked / num_total_masked}")
            print(f"Overall accuracy: {num_correct / num_total}")
            print('*' * 100)
            print()

if __name__ == '__main__':
    compression_scores()
    accuracy('outputs/bert-base-cased_poesia_1000_test')
