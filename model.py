import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math
import matplotlib.pyplot as plt

from data_set import DataSet, one_hot_encoding_table

class LatentPolynomialTransformer(nn.Module):
    # data_set automatically generates training and validation data given the content of "results/polynomial.json"
    # d_model is the embedding dimension of the transformer
    # nhead is the number of attention heads per layer in the transformer
    # num_layers is how many attention layers are in the transformer
    # dim_feedforward is the hidden layer width of the post attention feedforward network
    # dim_head is the width of the final prediction network
    # lr is the learning rate
    def __init__(self, data_set: DataSet, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dim_head, lr: float):
        super().__init__()

        self.pad_token = len(one_hot_encoding_table)

        vocab_size = len(one_hot_encoding_table) + 1
        max_input_seq_len = max(len(input_seq) for input_seq, _ in (data_set.training_data + data_set.validation_data))

        # switch this to query for cuda if running on unity
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        training_inputs = []
        training_expected_outputs = []
        for id_seq, expected_output in data_set.training_data:
            id_seq_tensor = torch.tensor(id_seq, dtype=torch.long, device=device)

            pad_len = max_input_seq_len - id_seq_tensor.size(0)
            padded_id_seq_tensor = torch.cat([id_seq_tensor, torch.full(tuple([ pad_len ]), self.pad_token, dtype=torch.long, device=device)])

            expected_output_tensor = torch.tensor([ expected_output ], dtype=torch.float32, device=device)

            training_inputs.append(padded_id_seq_tensor)
            training_expected_outputs.append(expected_output_tensor)

        self.training_data = TensorDataset(torch.stack(training_inputs), torch.stack(training_expected_outputs))

        validation_inputs = []
        validation_expected_outputs = []
        for id_seq, expected_output in data_set.validation_data:
            id_seq_tensor = torch.tensor(id_seq, dtype=torch.long, device=device)

            pad_len = max_input_seq_len - id_seq_tensor.size(0)
            padded_id_seq_tensor = torch.cat([id_seq_tensor, torch.full(tuple([ pad_len ]), self.pad_token, dtype=torch.long, device=device)])

            expected_output_tensor = torch.tensor([ expected_output ], dtype=torch.float32, device=device)

            validation_inputs.append(padded_id_seq_tensor)
            validation_expected_outputs.append(expected_output_tensor)

        self.validation_data = TensorDataset(torch.stack(validation_inputs), torch.stack(validation_expected_outputs))

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_input_seq_len + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, dim_head),
            nn.ReLU(),
            nn.Linear(dim_head, dim_head),
            nn.ReLU(),
            nn.Linear(dim_head, dim_head),
            nn.ReLU(),
            nn.Linear(dim_head, 1)
        )

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.to(device)

    def forward(self, inputs: torch.tensor):
        batch_size, seq_len = inputs.shape

        token_embeddings = self.token_emb(inputs)

        positions = torch.arange(seq_len, device=inputs.device).unsqueeze(0)
        positional_embeddings = self.pos_emb(positions)

        embeddings = token_embeddings + positional_embeddings

        mask = inputs == self.pad_token

        embeddings = self.encoder(embeddings, src_key_padding_mask=mask)

        pooled = embeddings.mean(dim=1)

        return self.head(pooled)
    
    def train(self, inputs: torch.tensor, expected_outputs: torch.tensor):
        self.optimizer.zero_grad()

        loss = self.loss_fn(self.forward(inputs), expected_outputs)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        self.optimizer.step()

        return loss
    
    def loss(self, inputs: torch.tensor, expected_outputs: torch.tensor):
        loss = self.loss_fn(self.forward(inputs), expected_outputs)
        
        return loss
    
def main():
    model = LatentPolynomialTransformer(DataSet(3600, 360), 128, 2, 4, 128, 128, 2e-5)

    epoch_samples = []
    training_loss_samples = []
    validation_loss_samples = []

    # view samples during training for empirical validation
    sample_inputs, sample_expected_outputs = model.validation_data.tensors
    sample_inputs = sample_inputs[:7]
    sample_expected_outputs = sample_expected_outputs[:7]

    train_loader = DataLoader(model.training_data, batch_size=128, shuffle=True)

    for i in range(0, 2000):
        for inputs, expected_outputs in train_loader: model.train(inputs, expected_outputs)

        training_inputs, training_expected_outputs = model.training_data.tensors
        training_loss = model.loss(training_inputs, training_expected_outputs)

        validation_inputs, validation_expected_outputs = model.validation_data.tensors
        validation_loss = model.loss(validation_inputs, validation_expected_outputs)

        epoch_samples.append(i)
        training_loss_samples.append(math.log10(training_loss.detach().cpu().numpy()))
        validation_loss_samples.append(math.log10(validation_loss.detach().cpu().numpy()))

        if i % 10 == 0:
            print(f"After Batch {i}:")
            print(f"\tTraining Loss: {training_loss}")
            print(f"\tValidation Loss: {validation_loss}")
            print(f"\tExpected: {sample_expected_outputs.squeeze(1)}")
            print(f"\tObserved: {model.forward(sample_inputs).squeeze(1)}\n")

    plt.plot(epoch_samples, training_loss_samples, label="Training Loss")
    plt.plot(epoch_samples, validation_loss_samples, label="Validation Loss")
    plt.xlim(left=0)
    plt.xlabel("Epoch")
    plt.ylabel("Log MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.show()

if __name__ == "__main__": main()