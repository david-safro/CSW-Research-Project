import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from qiskit import Aer
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.algorithms.optimizers import COBYLA

def main():
    print("Loading data...")
    data = pd.read_csv('data.csv')  # Replace with the path to actual file
    print("Data loaded successfully!")

    print("Starting data preprocessing...")
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols.remove('HeartDisease')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])
    X = data.drop('HeartDisease', axis=1)
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(data['HeartDisease'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    print("Data preprocessing completed!")

    print("Setting up Quantum Neural Network (QNN)...")
    quantum_instance = QuantumInstance(Aer.get_backend('aer_simulator'), shots=1024)

    feature_dimension = X_train_transformed.shape[1]
    feature_map = ZZFeatureMap(feature_dimension)
    ansatz = RealAmplitudes(feature_dimension, reps=1)
    qnn = TwoLayerQNN(feature_dimension, feature_map, ansatz, quantum_instance=quantum_instance)
    print("QNN setup completed!")

    print("Starting QNN training...")
    optimizer = COBYLA()
    classifier = NeuralNetworkClassifier(qnn, optimizer=optimizer)
    classifier.fit(X_train_transformed, y_train)
    print("QNN training completed!")

    return classifier, X_test_transformed, y_test

if __name__ == "__main__":
    classifier, X_test_transformed, y_test = main()
