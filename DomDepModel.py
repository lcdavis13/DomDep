import torch
import torch.nn as nn
import torch.nn.functional as F

class DomDepModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, blocks, embedding_dim):
        super(DomDepModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.blocks = blocks
        self.embedding_dim = embedding_dim

        # Define the embedding layer
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim) # +1 for zero padding/index

        # Adjust the input size for the first feedforward layer to match embedding dimension
        self.ff1 = nn.Linear(embedding_dim, hidden_size)
        self.ff2 = nn.Linear(hidden_size, output_size)
        self.mod_factor_gen = nn.Linear(hidden_size, blocks)

        assert hidden_size % blocks == 0, "hidden_size must be divisible by blocks"

    def forward(self, x):
        # Pass inputs through the embedding layer
        x = self.embedding(x)

        # Assuming the input x is of shape (batch_size, sequence_length), we need to handle it
        # properly before passing through the linear layers. One approach could be to average the embeddings
        # or simply take the last one. Here, we'll average the embeddings to maintain the dimensionality.
        x = torch.mean(x, dim=1)

        x = F.relu(self.ff1(x))
        mod_factors_pre_softmax = self.mod_factor_gen(x)
        mod_factors = F.softmax(mod_factors_pre_softmax, dim=1)
        block_size = self.hidden_size // self.blocks
        mod_factors = mod_factors.repeat_interleave(block_size, dim=1)
        mod_factors = mod_factors[:, :self.hidden_size]
        x = x * mod_factors
        x = self.ff2(x)
        return x


if __name__ == "__main__":
    vocab_size = 10  # Assuming a vocabulary size of 10 for the example
    hidden_size = 100
    output_size = 10  # Assuming output size matches vocab size for simplicity
    blocks = 10
    embedding_dim = 20  # Size of the embedding vectors

    # Initialize the model with the new parameters
    model = DomDepModel(vocab_size, hidden_size, output_size, blocks, embedding_dim)

    # Create a dummy input tensor of integer indices
    # For this example, let's simulate a batch of 1 with a sequence of 5 indices
    dummy_input = torch.randint(low=1, high=vocab_size + 1, size=(1, 5), dtype=torch.long)

    # Forward pass
    output = model(dummy_input)
    print(output)
    print(output.shape)