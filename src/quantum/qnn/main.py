import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from qiskit import Aer
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.algorithms.optimizers import COBYLA

print("Loading data...")
data = pd.read_csv('data.csv')  # Replace with the path to actual file
print("Data loaded successfully!")

print("Selecting a subset of 10 rows from the dataset...")
subset_data = data.sample(n=918, random_state=1)
print("Subset selected successfully!")

print("Starting data preprocessing...")
categorical_cols = subset_data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = subset_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('HeartDisease')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])
X_subset = subset_data.drop('HeartDisease', axis=1)
y_encoder_subset = LabelEncoder()
y_subset = y_encoder_subset.fit_transform(subset_data['HeartDisease'])
X_train_subset, X_test_subset, y_train_subset, y_test_subset = train_test_split(
    X_subset, y_subset, test_size=0.2, random_state=1
)
X_train_transformed_subset = preprocessor.fit_transform(X_train_subset)
X_test_transformed_subset = preprocessor.transform(X_test_subset)
print("Data preprocessing completed!")

print("Setting up Quantum Neural Network (QNN)...")
quantum_instance = QuantumInstance(Aer.get_backend('aer_simulator'), shots=1024)

feature_dimension = X_train_transformed_subset.shape[1]
feature_map = ZZFeatureMap(feature_dimension)
ansatz = RealAmplitudes(feature_dimension, reps=1)
qnn = TwoLayerQNN(feature_dimension, feature_map, ansatz, quantum_instance=quantum_instance)
print("QNN setup completed!")

print("Starting QNN training...")
optimizer = COBYLA()
classifier = NeuralNetworkClassifier(qnn, optimizer=optimizer)
classifier.fit(X_train_transformed_subset, y_train_subset)
print("QNN training completed!")

print("Making predictions with trained QNN...")
y_pred = classifier.predict(X_test_transformed_subset)
print("Predictions made successfully!")

accuracy = accuracy_score(y_test_subset, y_pred)
print(f'Accuracy of the QNN is: {accuracy * 100:.2f}%')