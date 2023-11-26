import torch
from torch import nn
from data import data_preprocessing
import pickle
from sklearn.metrics import accuracy_score as accuracy
from constants import *

#PICKLE
def model_weights (model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model.state_dict(), f)
def load_model_weights(model, filename):
    with open(filename, 'rb') as f:
        model.load_state_dict(pickle.load(f))

# data preprocessing
X_train, X_test, y_train, y_test, scaler = data_preprocessing(CSV_FILE)

#device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_train, y_train = torch.tensor(X_train, dtype = torch.float32).to(device), torch.tensor(y_train, dtype = torch.float32).to(device)
X_test, y_test = torch.tensor(X_test, dtype = torch.float32).to(device), torch.tensor(y_test, dtype = torch.float32).to(device)



#neural network architecture
class NeuralNetwork (nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features = 11, out_features = 300)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features = 300, out_features = 100)
        self.fc3 = nn.Linear(in_features = 100 , out_features = 1)
        self.bn  = nn.BatchNorm1d(300)
        self.dropout = nn.Dropout()
    def forward (self,x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


model = NeuralNetwork().to(device)
load_model_weights(model, 'model_weights.pkl')
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(params = model.parameters(), lr = LR)
torch.manual_seed(42)

#train-test loop
model.eval()
with torch.no_grad():
    test_logits = model(X_train).squeeze()
    test_loss = loss_fn(test_logits, y_train)
#Accuracy on the test set
train_final = torch.round(torch.sigmoid(test_logits)).cpu().detach().numpy()
train_true = y_train.cpu().detach().numpy()
accuracy = accuracy(train_true, train_final)
print("accuracy on the train set " + str(accuracy*100))
