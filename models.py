import constants
import torch
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
def results(y_logit, y_true, tokenizer, compression_ids):
    y_pred = compression_ids[F.softmax(y_logit, 1).argmax(dim=1)]
    y_true = compression_ids[y_true]
    with open('results.txt', 'a') as f:
        f.write('-' * 100 + '\n')
        f.write(f"{tokenizer.decode(y_pred)}\n\n")
        f.write(f"{tokenizer.decode(y_true)}\n")
    return torch.sum(y_pred == y_true).item(), y_true.numel()

# TODO remove dependency on tokenizer
def train(model, tokenizer, dataloader, compression_ids, training_args):
    optimizer = training_args['optimizer'](model.parameters(), training_args['lr'])
    loss_fn = training_args['loss_fn']

    model.train()
    for i in range(training_args['epochs']):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        for batch in tqdm(dataloader):
            model_inputs, true_tokens, mask_indices = batch
            optimizer.zero_grad()
            preds = model(model_inputs)
            loss = loss_fn(preds[mask_indices], true_tokens[mask_indices])
            batch_correct, batch_total = results(preds[mask_indices], true_tokens[mask_indices], tokenizer, compression_ids)
            epoch_correct += batch_correct
            epoch_total += batch_total
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {i}:\nLoss: {epoch_loss}\nAccuracy: {epoch_correct / epoch_total}")
        print('-' * 100)

def predict(model, tokenized_txt):
    model.eval()
