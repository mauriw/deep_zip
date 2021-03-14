import constants
import torch

"""
Given input masked (compressed) text, returns the decompressed text

Arguments:
    masked_txt: masked/compressed input
    model: model used for compression
    tokenizer: tokenizer used for parsing text
"""
def decompress(masked_txt, model, tokenizer):
    inputs = tokenizer(masked_txt, return_tensors='pt') # tokenize input
    inputs.to(constants.DEVICE)
    # This is to ensure that we only predict on [MASK] tokens later
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    outputs = model(**inputs)

    # Predict
    mask_preds = []
    for i, token in enumerate(tokens):
        if token == constants.MASK:
            mask_preds.append(tokenizer.convert_ids_to_tokens(torch.argmax(outputs.logits[0, i]).item()))

    # Reconstitute
    mask_idx = 0
    split_txt = masked_txt.split()
    for i, word in enumerate(split_txt):
        if word == constants.MASK:
            split_txt[i] = mask_preds[mask_idx]
            mask_idx += 1
    return ' '.join(split_txt)

"⁇"
