import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

from data_set import DataSet, one_hot_encoding_table

class LatentPolynomialTransformer(nn.Module):
    def __init__(self, data_set: DataSet, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dim_head, lr: float):
        super().__init__()
        
        self.data_set = data_set

        self.pad_token = len(one_hot_encoding_table)

        vocab_size = len(one_hot_encoding_table) + 1
        max_input_seq_len = max(len(input_seq) for input_seq, _ in (data_set.training_data + data_set.validation_data))

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.training_tensors = []
        for id_seq, expected_output in data_set.training_data:
            id_seq_tensor = torch.tensor(id_seq, dtype=torch.long, device=device)

            pad_len = max_input_seq_len - id_seq_tensor.size(0)
            padded_id_seq_tensor = torch.cat([id_seq_tensor, torch.full(tuple([ pad_len ]), self.pad_token, dtype=torch.long, device=device)])

            expected_output_tensor = torch.tensor([ expected_output ], dtype=torch.float32, device=device)

            self.training_tensors.append((padded_id_seq_tensor, expected_output_tensor))

        self.validation_tensors = []
        for id_seq, expected_output in data_set.validation_data:
            id_seq_tensor = torch.tensor(id_seq, dtype=torch.long, device=device)

            pad_len = max_input_seq_len - id_seq_tensor.size(0)
            padded_id_seq_tensor = torch.cat([id_seq_tensor, torch.full(tuple([ pad_len ]), self.pad_token, dtype=torch.long, device=device)])

            expected_output_tensor = torch.tensor([ expected_output ], dtype=torch.float32, device=device)

            self.validation_tensors.append((padded_id_seq_tensor, expected_output_tensor))

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
            nn.Linear(dim_head, dim_head),
            nn.ReLU(),
            nn.Linear(dim_head, 1)
        )

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.to(device)

    def forward(self, input_ids: torch.tensor):
        batch_size, seq_len = input_ids.shape

        token_embeddings = self.token_emb(input_ids)

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positional_embeddings = self.pos_emb(positions)

        embeddings = token_embeddings + positional_embeddings

        mask = input_ids == self.pad_token

        embeddings = self.encoder(embeddings, src_key_padding_mask=mask)

        pooled = embeddings.mean(dim=1)

        return self.head(pooled)
    
    def train(self, training_tensors: list[(torch.tensor, torch.tensor)]):
        input_ids = []
        expected_outputs = []

        for id_seq, expected_output in training_tensors:
            input_ids.append(id_seq)
            expected_outputs.append(expected_output)

        input_ids = torch.stack(input_ids)
        expected_outputs = torch.stack(expected_outputs)

        self.optimizer.zero_grad()

        loss = self.loss_fn(self.forward(input_ids), expected_outputs)

        loss.backward()

        self.optimizer.step()

        return loss
    
    def loss(self, validation_tensors: list[(torch.tensor, torch.tensor)]):
        input_ids = []
        expected_outputs = []

        for id_seq, expected_output in validation_tensors:
            input_ids.append(id_seq)
            expected_outputs.append(expected_output)

        input_ids = torch.stack(input_ids)
        expected_outputs = torch.stack(expected_outputs)

        return self.loss_fn(self.forward(input_ids), expected_outputs)
    
def main():
    model = LatentPolynomialTransformer(DataSet(1200, 300), 64, 2, 4, 64, 64, 3e-3)

    epoch_samples = []
    training_loss_samples = []
    validation_loss_samples = []

    sample_input_ids = []
    sample_expectations = []

    for input_ids, expected_output in model.validation_tensors[0:7]:
        sample_input_ids.append(input_ids)
        sample_expectations.append(expected_output)
    
    sample_input_ids = torch.stack(sample_input_ids)
    sample_expectations = torch.stack(sample_expectations)

    for i in range(0, 2000):
        training_loss = model.train(model.training_tensors)
        validation_loss = model.loss(model.validation_tensors)

        epoch_samples.append(i)
        training_loss_samples.append(math.log10(training_loss.detach().cpu().numpy()))
        validation_loss_samples.append(math.log10(validation_loss.detach().cpu().numpy()))

        if i % 10 == 0:
            print(f"After Batch {i}:")
            print(f"\tTraining Loss: {training_loss}")
            print(f"\tValidation Loss: {validation_loss}")
            print(f"\tExpected: {sample_expectations.squeeze(1)}")
            print(f"\tObserved: {model.forward(sample_input_ids).squeeze(1)}\n")

    plt.plot(epoch_samples, training_loss_samples, label="Training Loss")
    plt.plot(epoch_samples, validation_loss_samples, label="Validation Loss")
    plt.xlim(left=0)
    plt.xlabel("Epoch")
    plt.ylabel("Log MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.show()

if __name__ == "__main__": main()