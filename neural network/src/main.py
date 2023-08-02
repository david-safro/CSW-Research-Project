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

model = NeuralNetwork().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
torch.manual_seed(42)

# Train and test loop
for epoch in range(EPOCH_COUNT):
    model.train()
    logits = model(X_train).squeeze()
    final = torch.round(torch.sigmoid(logits))
    loss = loss_fn(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test).squeeze()
        test_final = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{EPOCH_COUNT}, Loss: {loss.item():.4f}')

# Calculate accuracy on the test set
y_pred = torch.round(torch.sigmoid(test_logits)).cpu().detach().numpy()
y_true = y_test.cpu().detach().numpy()
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy on the test set: {accuracy:.4f}')

# Save the model weights and biases
save_model_weights(model, 'model_weights.pkl')
