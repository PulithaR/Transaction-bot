# Import necessary libraries
import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Import custom utilities and model
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Load intents from a intents.json file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Initialize lists to store words, tags, and patterns
all_words = []
tags = []
xy = []

# Process intents and extract words and tags
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Stem and lower each word, remove duplicates, and sort
Stop_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in Stop_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Hyper-parameters
num_epochs = 3000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 16  # Increased hidden size
output_size = len(tags)

# Define a custom dataset for PyTorch
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Create DataLoader for training and validation sets
train_dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_dataset = ChatDataset(X_val, y_val)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Use GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the neural network model
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Validate the model every 100 epochs
    if (epoch+1) % 100 == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (words, labels) in val_loader:
                words = words.to(device)
                labels = labels.to(dtype=torch.long).to(device)
                outputs = model(words)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        model.train()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

# Save the trained model and related information
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete. Model saved to {FILE}')
