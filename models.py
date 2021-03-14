import constants
import torch
import torch.nn.functional as F

class BertFinetune(torch.nn.Module):
    def __init__(self, encoder, lm_output_size, output_vocab_size):
        super(BertFinetune, self).__init__()
        self.encoder = encoder
        self.encoder.to(constants.DEVICE)
        self.output_vocab_size = output_vocab_size

        # Module vars
        # TODO see what difference bias makes
        self.lin = torch.nn.Linear(lm_output_size, output_vocab_size, bias=False)

    def forward(self, inputs):
        output_hidden_state = self.encoder(**inputs).last_hidden_state
        x = self.lin(output_hidden_state)
        return F.softmax(x, -1)

def train(model, tokenizer, training_args, masked_txts, true_txts):
    optimizer = training_args['optimizer']
    loss_fn = training_args['loss_fn']

    model_inputs = tokenizer(masked_txts, padding=True, return_tensors='pt')
    true_tokens = tokenizer(true_txts, padding=True, return_tensors='pt').input_ids
    mask_indices = model_inputs.input_ids == tokenizer.mask_token_id

    # print(mask_indices)
    # print(mask_indices.shape)

    model.train()
    for _ in range(training_args['epochs']):
        optimizer.zero_grad()
        preds = model(model_inputs)
        # print(preds.shape)
        # print(true_tokens.shape)
        print(preds[mask_indices].argmax(dim=1))
        print(true_tokens[mask_indices])
        print(tokenizer.decode(preds[mask_indices].argmax(dim=1)))
        loss = loss_fn(preds[mask_indices], true_tokens[mask_indices])
        loss.backward()
        optimizer.step()
        print(loss.item())

def predict(model, tokenized_txt):
    model.eval()
