import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MASK = '[MASK]'
NEW_MASK = '~~'
NEW_MASK_ID = -1
PRETRAINED_BERT_MAX_LEN = 512
PRETRAINED_BERT_OUTPUT_HIDDEN_SIZE = 768
