import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class AssociativeRecallDataset(Dataset):
    def __init__(self, sequence_length, vocab_size, dataset_size=10000):
        assert sequence_length > 2, "Sequence length must be greater than 2."
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.dataset_size = dataset_size
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        # Generate a non-ambiguous sequence
        # First, generate a sequence where each item (except the last one) is unique
        unique_sequence_length = self.sequence_length - 1
        sequence = np.random.choice(range(1, self.vocab_size + 1), size=unique_sequence_length, replace=False)
        
        # Select a random item from the sequence (except the last one) to duplicate
        recall_cue_index = np.random.randint(0, unique_sequence_length - 1)
        recall_cue = sequence[recall_cue_index]
        
        # The target is the item following the recall cue
        target = sequence[recall_cue_index + 1]
        
        # Append the recall cue to the end of the sequence
        input_sequence_with_cue = np.append(sequence, recall_cue)
        
        # Convert to tensors
        input_tensor = torch.tensor(input_sequence_with_cue, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)
        
        return input_tensor, target_tensor

if __name__ == "__main__":
    # Parameters
    sequence_length = 30  # This includes the recall cue at the end
    vocab_size = 100  # Number range for symbols
    dataset_size = 1000  # Size of the dataset
    
    # Create the dataset and DataLoader
    dataset = AssociativeRecallDataset(sequence_length, vocab_size, dataset_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Example of iterating over the DataLoader (in a training loop)
    for input_sequence, target in dataloader:
        print(f"Input Sequence: {input_sequence}")
        print(f"Target: {target}")
        # Add your model training code here
        break  # Stop after one iteration