import torch

DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
PRETRAINED_BERT_MAX_LEN = 512
