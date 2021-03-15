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

    def compress(self, text):
        toks = text.split()
        comp = [word if word not in self.compression_words else \
                        constants.MASK for word in toks]
        return ' '.join(comp)

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
        text = self.dataset[index]
        compression = self.compress(text)
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
def get_dataloader(compressor, **kwargs):
    dataset = CompressedDataset(compressor)
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
