import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from a JSON file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load pre-trained model and related data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize and load the neural network model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def get_response(msg):
    sentence = tokenize(msg)
    print(f"Tokenized sentence: {sentence}")

    X = bag_of_words(sentence, all_words)
    print(f"Bag of words: {X}")

    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    print(f"Predicted tag: {tag}")

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(f"Probability: {prob.item()}")

    if prob.item() > 0.5:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break

        resp = get_response(sentence)
        print(resp)
