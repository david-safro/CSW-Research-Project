import torch
import pickle

from torch import nn

from data import read_and_preprocess_data
from neural_network import NeuralNetwork
from sklearn.metrics import accuracy_score
from constants import *

def save_model_weights(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model.state_dict(), f)
def load_model_weights(model, filename):
    with open(filename, 'rb') as f:
        model.load_state_dict(pickle.load(f))
# Read and preprocess the data
X_train, X_test, y_train, y_test, scaler = read_and_preprocess_data(CSV_FILE)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

# Model setup
input_size = INPUT_SIZE
hidden_size = HIDDEN_SIZE
output_size = OUTPUT_SIZE


loaded_model = NeuralNetwork().to(device)
load_model_weights(loaded_model, 'model_weights.pkl')
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(loaded_model.parameters(), lr=LEARNING_RATE)
torch.manual_seed(42)

loaded_model.eval()
with torch.inference_mode():
    train_logits = loaded_model(X_train).squeeze()
    train_final = torch.round(torch.sigmoid(train_logits))
    train_loss = loss_fn(train_logits, y_train)

# Calculate accuracy on the training set
train_pred = torch.round(torch.sigmoid(train_logits)).cpu().detach().numpy()
train_accuracy = accuracy_score(y_train.cpu().detach().numpy(), train_pred)
print(f'Accuracy on the training set: {train_accuracy:.4f}')