import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MASK = '[MASK]'
PRETRAINED_BERT_MAX_LEN = 512
TRAIN = 0.8
VAL = 0.1
TEST = 0.1
