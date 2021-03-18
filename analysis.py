import gzip
import collections
import plotly.graph_objects as go

import constants
import data
import compressor

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

if __name__ == '__main__':
    compression_scores()
