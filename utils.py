import datasets

"""
Returns raw wikitext train input
"""
def get_data():
    # return wikitext dataset
    dataset = datasets.load_dataset('wikitext', 'wikitext-2-v1')
    return dataset['train']['text']
