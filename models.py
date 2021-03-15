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

def train(model, tokenizer, dataloader, training_args):
    optimizer = training_args['optimizer']
    loss_fn = training_args['loss_fn']

    model.train()
    for i in range(training_args['epochs']):
        epoch_loss = 0
        for batch in tqdm(dataloader):
            model_inputs, true_tokens = batch
            mask_indices = model_inputs['input_ids'] == tokenizer.mask_token_id
            optimizer.zero_grad()
            preds = model(model_inputs)
            loss = loss_fn(preds[mask_indices], true_tokens[mask_indices])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {i}: {epoch_loss}")
        print('-' * 100)

def predict(model, tokenized_txt):
    model.eval()
