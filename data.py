import constants
import datasets
from torch.utils.data import DataLoader, Dataset
from compressor import preprocess_tf_idf


"""
Class Defintion:
Compression dataset generates compression/original text pairs
from the wikitext HuggingFace dataset
"""
class CompressedDataset(Dataset):
    def __init__(self, idf=True):
        self.dataset = datasets.load_dataset('wikitext', \
                                'wikitext-2-v1')['train']['text']

        # preprocessing for compressor
        if idf:
            self.compression_words = preprocess_tf_idf()
        else:
            self.compression_words = None # TODO: add Gabriel's

    def compress(self, text):
        toks = text.split()
        comp = [word if word not in self.compression_words else \
                        constants.MASK for word in toks]
        return ' '.join(comp)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index : int):
        """
        Function called by Torch to get an element.
        At the moment, empty strings are not filter out (TODO)
        Args:
            index: index of elem to retrieve
        """
        text = self.dataset[index]
        compression = self.compress(text)
        return compression, text


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
