import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MASK = '[MASK]'
PRETRAINED_BERT_MAX_LEN = 512