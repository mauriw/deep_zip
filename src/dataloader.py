import datasets

def get_data():
    dataset = datasets.load_dataset('wikitext', 'wikitext-2-v1')
    return dataset['train']['text']
