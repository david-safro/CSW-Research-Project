import sys
import os
import torch
import torch.nn as nn
from constants import *
# Add the parent directory of the neural network folder to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data import HeartDiseasePrediction
csv_file = 'Heart_Disease_Prediction.csv'
prediction = HeartDiseasePrediction()
prediction.read_data()
prediction.preprocess_data()
prediction.split_data()

# Access the attributes or call other methods as needed
print("Sample data:")
print(prediction.data.head())

print("Scaled features:")
print(prediction.X_scaled[:5])

print("Training set shape"":", prediction.X_train.shape, prediction.y_train.shape)
print("Testing set shape:", prediction.X_test.shape, prediction.y_test.shape)

device = "cuda" if torch.cuda.is_available() else "cpu"
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


X_train, y_train = prediction.X_train.to(device), prediction.y_train.to(device)
X_test, y_test = prediction.X_test.to(device), prediction.y_test.to(device)


def main():
    model = NeuralNetwork.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params = NeuralNetwork.parameters(), lr=LR)
    torch.manual_seed(42)
    # train and test loop
    for epoch in range(EPOCH_COUNT):
        model.train()
        logits = model(X_train).squeeze()
        final = torch.round(torch.sigmoid(logits))
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test).squeeze()
            test_final = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{EPOCH_COUNT}, Loss: {loss_fn.item():.4f}')


if __name__ == '__main':
    main()
