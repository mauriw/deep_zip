import decompressor

from transformers import BertTokenizer, BertForMaskedLM

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertForMaskedLM.from_pretrained('bert-base-cased')
    print(decompressor.decompress("Hi there, [MASK] to [MASK] you!", model, tokenizer))