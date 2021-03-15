import constants
import datasets
from torch.utils.data import DataLoader, Dataset
from compressor import preprocess_tf_idf, preprocess_poesia
import pickle as pkl
from pathlib import Path

"""
Class Defintion:
Compression dataset generates compression/original text pairs
from the wikitext HuggingFace dataset
"""
class CompressedDataset(Dataset):
    def __init__(self, idf=True, vocab_size=10000, split="train", compression_words=None):
        self.vocab_size = vocab_size
        self.dataset = get_dataset(split)
        
        if compression_words:
            self.compression_words = compression_words
        if idf:
            self.compression_words = preprocess_tf_idf(self.dataset, vocab_size)
        else:
            self.compression_words = preprocess_poesia(self.dataset, vocab_size)

        self.compression_indices = self.tokenizer.convert_tokens_to_ids(self.compression_words)

    def preprocess(self, text):
        return text.strip().replace('<unk>', '[UNK]')

    def compress(self, text):
        toks = self.tokenizer.tokenize(text)
        comp = [constants.MASK if word in self.compression_words else \
                word for word in toks]
        return self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(comp))

    def prediction_size(self):
        return self.vocab_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index : int):
        """
        Function called by Torch to get an element.
        Args:
            index: index of elem to retrieve
        """
        compression = self.tokenizer(self.preprocess(self.dataset[index]), padding='max_length',
                                     max_length=constants.PRETRAINED_BERT_MAX_LEN, return_tensors='pt',
                                     truncation=True)
        text = compression['input_ids'].clone().squeeze().to(constants.DEVICE)
        for i in range(compression['input_ids'].shape[-1]):
            if compression['input_ids'][0, i].item() in self.compression_indices:
                compression['input_ids'][0, i] = self.tokenizer.mask_token_id
        compression['input_ids'] = compression['input_ids'].squeeze().to(constants.DEVICE)
        compression['token_type_ids'] = compression['token_type_ids'].squeeze().to(constants.DEVICE)
        compression['attention_mask'] = compression['attention_mask'].squeeze().to(constants.DEVICE)
        return compression, text

"""
Returns cleaned dataset

Arguments:
    split: string, "train", "val", or "test"
"""
def get_dataset(split):
    assert split == 'train' or split == 'val' or split == 'test'
    split = 'validation' if split == 'val' else split
    filename = Path.cwd() / ('wikitext2_clean_' + split + '.p')
    if filename.exists():
        with open(filename, 'rb') as f:
            return pkl.load(f)
    
    ds = datasets.load_dataset('wikitext', 'wikitext-2-v1')[split]['text']
    clean = []
    for d in ds:
        if not d: 
            continue
        if d.startswith(' ='):
            continue
        if len(d.split()) < 10:
            continue
        d = d.replace('<unk>', '[UNK]')
        d = d.replace(' @-@ ', '-')
        d = d.replace(' @,@ ', ',')
        d = d.replace(' @.@ ', '.')
        clean.append(d)
    
    print('Writing cleaned dataset to', filename.name)
    with open(filename, 'wb') as f:
        x = pkl.dump(clean, f)     
    return clean

"""
Returns torch dataloader that for a dataset that
utilizes the specified compressor.

Arguments:
    compressor: bool, True if using TF-IDF (only IDF atm)
    kwargs: configurations for the torch DataLoader class
"""
def get_dataloader(tokenizer, compressor, **kwargs):
    dataset = CompressedDataset(tokenizer, compressor)
    return DataLoader(dataset, **kwargs)

"""
Test Harness
"""
def test(n_iter):
    TF_IDF = True
    data_loader = get_dataloader(TF_IDF, shuffle=True, batch_size=1)
    print("LEN TEST: \n", len(data_loader))
    print("ITEM TEST: \n")
    for i, data in enumerate(data_loader):
        print(data)
        if  i == n_iter: break
    # iter test:
    tr_it = iter(data_loader)
    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(train_dataloader)
        data = next(tr_it)


if __name__ == "__main__":
    test(6)
