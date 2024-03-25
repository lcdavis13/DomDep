import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from AssociativeRecallDataset import AssociativeRecallDataset
from DomDepModel import DomDepModel


# Parameters
sequence_length_power = 5
vocab_size = 20
dataset_size = 10000
sequence_length = 2 ** sequence_length_power

holdout_ratio = 0.2
test_size = int(dataset_size * holdout_ratio)
train_size = dataset_size - test_size

batch_size = 64

embedding_dim = 50  # Dimension of the embedding vector
hidden_size = 128
output_size = vocab_size + 1  # Assuming the output size is the same as vocab size, plus one for zero padding/index
blocks = 4
learning_rate = 0.001
epochs = 10

# Dataset
dataset = AssociativeRecallDataset(sequence_length, vocab_size, dataset_size)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
model = DomDepModel(vocab_size, hidden_size, output_size, blocks, embedding_dim)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Training
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs)  # inputs are now correctly processed through the embedding layer
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader)}, Accuracy: {train_accuracy}%')

# Testing
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)  # Ensure inputs go through the embedding
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f'Test Loss: {test_loss / len(test_loader)}, Test Accuracy: {test_accuracy}%')
