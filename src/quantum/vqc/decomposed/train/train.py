import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from qiskit_machine_learning.algorithms import QSVC
import joblib  # For saving the model

def data_preprocessing(csv):
    data = pd.read_csv(csv)
    data['Sex'] = data['Sex'].map({'M' : 1, 'F' : 0}).values
    data['ChestPainType'] = data['ChestPainType'].map({'TA' : 0, 'ATA' : 1, 'NAP' : 2, 'ASY' : 3}).values
    data['RestingECG'] = data['RestingECG'].map({'Normal' : 0, 'ST' : 1, 'LVH' : 2}).values
    data['ExerciseAngina'] = data['ExerciseAngina'].map({'Y' : 1, 'N' : 0})
    data['ST_Slope'] = data['ST_Slope'].map({'Up' : 0, 'Flat' : 1, 'Down':2})
    X = data.drop(columns = ['HeartDisease'])
    y = data["HeartDisease"].values
    scaler = StandardScaler()
    X= scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)
    return X_train, X_test, y_train, y_test
def train_model(X_train, y_train):
    qsvc = QSVC()
    print("Training started")
    qsvc.fit(X_train, y_train)
    return qsvc

def main():
    X_train, X_test, y_train, y_test = data_preprocessing('Heart_Disease_Prediction.csv')
    qsvc = train_model(X_train, y_train)
    joblib.dump(qsvc, 'qsvc_model.pkl')  # Save the model
    joblib.dump((X_test, y_test), 'test_data.pkl')  # Save test data

if __name__ == "__main__":
    main()
