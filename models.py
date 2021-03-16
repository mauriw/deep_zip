import constants
import torch

import pandas as pd
import plotly.express as px
import torch.nn.functional as F

from tqdm import tqdm

class BertFinetune(torch.nn.Module):
    def __init__(self, encoder, lm_output_size, output_vocab_size):
        super(BertFinetune, self).__init__()
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.output_vocab_size = output_vocab_size

        # Module vars
        # TODO see what difference bias makes
        self.lin = torch.nn.Linear(lm_output_size, output_vocab_size, bias=False)

        self.to(constants.DEVICE)

    # returns logits
    def forward(self, inputs):
        output_hidden_state = self.encoder(**inputs).last_hidden_state
        return self.lin(output_hidden_state)

# Returns num_correct, num_total
# def results(input, y_logit, y_true, mask_indices, tokenizer, compression_ids):
#     y_pred_tot = y_logit.detach().clone()
#     print(y_pred_tot[mask_indices].shape)
#     print((compression_ids[F.softmax(y_logit[mask_indices], 1).argmax(dim=1)]).shape)
#     y_pred_tot[mask_indices] = compression_ids[F.softmax(y_logit[mask_indices], 1).argmax(dim=1)]
#     y_true_tot = y_true.detach().clone()
#     y_true_tot[mask_indices] = compression_ids[y_true[mask_indices]]
#     # y_pred_mask = compression_ids[F.softmax(y_logit[mask_indices], 1).argmax(dim=1)]
#     # y_true_mask = compression_ids[y_true[mask_indices]]
#     with open('results.txt', 'a') as f:
#         f.write('-' * 100 + '\n')
#         f.write(f"{tokenizer.decode(y_pred_tot)}\n\n")
#         f.write(f"{tokenizer.decode(y_true_tot)}\n")
#     return torch.sum(y_pred_tot == y_true_tot).item(), y_true_tot.numel()

def train(model, tokenizer, train_dset, val_dset, compression_ids, run_name, training_args):
    print(f"Training {run_name}:")

    optimizer = training_args['optimizer'](model.parameters(), training_args['lr'])
    loss_fn = training_args['loss_fn']

    best_val_accuracy = 0
    history = pd.DataFrame()
    for i in range(training_args['epochs']):
        print('-' * 100)
        print(f"Epoch {i}:\n")
        print("Train:")
        train_loss, train_accuracy, _ = run(model, tokenizer, train_dset, compression_ids, f'{run_name}_train_{i}', loss_fn, train=True, optimizer=optimizer)
        print(f"Train Loss: {train_loss}")
        print(f"Train accuracy: {train_accuracy}\n")

        print("Validation:")
        with torch.no_grad():
            val_loss, val_accuracy, _ = run(model, tokenizer, val_dset, compression_ids, f'{run_name}_val_{i}', loss_fn, train=False)
        print(f"Val Loss: {val_loss}")
        print(f"Val accuracy: {val_accuracy}")
        if val_accuracy > best_val_accuracy:
            print("New best validation accuracy achieved, saving model")
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f'best_models/{run_name}.params')

        history = history.append({'Epoch': i, 'Loss': train_loss, 'Accuracy': train_accuracy, 'Split': 'Train'}, ignore_index=True)
        history = history.append({'Epoch': i, 'Loss': val_loss, 'Accuracy': val_accuracy, 'Split': 'Valdation'}, ignore_index=True)
    for metric in ['Loss', 'Accuracy']:
        fig = px.line(history, x='Epoch', y=metric, color='Split')
        fig.update_xaxes(nticks=training_args['epochs'])
        fig.update_layout(title={
            'text': f"{metric} Over Epoch For {run_name}",
            'x': 0.46,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        })
        fig.write_image(f'plots/{run_name}_{metric.lower()}.png')

def eval(model, tokenizer, test_dset, compression_ids, run_name, loss_fn):
    print(f"Evaluating {run_name}:")
    loss, accuracy, tot_accuracy = run(model, tokenizer, test_dset, compression_ids, f'{run_name}_test', loss_fn, train=False)
    compression_rate = compression(f"{run_name}_test")
    summary = f"Loss: {loss}\nMasked accuracy: {accuracy}\nOverall accuracy: {tot_accuracy}\nCompression rate: {compression_rate}"
    print(summary)
    with open(f'results/{run_name}.txt', 'w') as f:
        f.write(f"{summary}\n")

def run(model, tokenizer, dset, compression_ids, output_fname, loss_fn, train=False, optimizer=None):
    if train:
        assert optimizer is not None, "If training, must specify an optimizer"

    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0
    epoch_tot_correct = 0
    epoch_tot_total = 0
    for batch in tqdm(dset):
        model_inputs, true_tokens, mask_indices = batch
        model_inputs['input_ids'][mask_indices] = tokenizer.mask_token_id
        if train:
            optimizer.zero_grad()
        preds = model(model_inputs)
        loss = loss_fn(preds[mask_indices], true_tokens[mask_indices])
        batch_correct, batch_total, tot_correct, tot_total = results(output_fname, model_inputs['input_ids'], preds, true_tokens,
                                                                     mask_indices, tokenizer, compression_ids)
        epoch_correct += batch_correct
        epoch_total += batch_total
        epoch_tot_correct += tot_correct
        epoch_tot_total += tot_total
        if train:
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(dset)
    epoch_accuracy = epoch_correct / epoch_total
    epoch_tot_accuracy = epoch_tot_correct / epoch_tot_total
    return epoch_loss, epoch_accuracy, epoch_tot_accuracy

def results(output_fname, input_tokens, y_logit, y_true, mask_indices, tokenizer, compression_ids):
    pred_tokens = input_tokens.detach().clone()
    y_true_tokens = y_true.detach().clone()
    pred_tokens[mask_indices] = compression_ids[F.softmax(y_logit[mask_indices], 1).argmax(dim=1)]
    y_true_tokens[mask_indices] = compression_ids[y_true[mask_indices]]

    clean = lambda s: s.replace(constants.MASK, constants.NEW_MASK).replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').strip()

    with open(f'outputs/{output_fname}_masked.txt', 'a') as f:
        for input_token in input_tokens:
            f.write(f"{clean(tokenizer.decode(input_token))}\n\n")
    with open(f'outputs/{output_fname}_preds.txt', 'a') as f:
        for pred in pred_tokens:
            f.write(f"{clean(tokenizer.decode(pred))}\n\n")
    with open(f'outputs/{output_fname}_true.txt', 'a') as f:
        for y_true in y_true_tokens:
            f.write(f"{clean(tokenizer.decode(y_true))}\n\n")
    return (torch.sum(pred_tokens[mask_indices] == y_true_tokens[mask_indices]).item(), mask_indices.count_nonzero(),
            torch.sum(pred_tokens == y_true_tokens).item(), mask_indices.numel())

def compression(base_fname):
    compressed_len = 0
    original_len = 0
    with open(f"outputs/{base_fname}_masked.txt") as f:
        for line in f:
            if not line.strip():
                continue
            compressed_len += len(line.encode('utf8'))
    with open(f"outputs/{base_fname}_true.txt") as f:
        for line in f:
            if not line.strip():
                continue
            original_len += len(line.encode('utf8'))
    return compressed_len / original_len
