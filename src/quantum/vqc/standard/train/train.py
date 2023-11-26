from qiskit.algorithms.optimizers import COBYLA, ADAM, SPSA
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, TwoLocal
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.primitives import Sampler
import time as t
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def data_preprocessing(csv):
    data = pd.read_csv(csv)
    data['Sex'] = data['Sex'].map({'M': 1, 'F': 0}).values
    data['ChestPainType'] = data['ChestPainType'].map({'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}).values
    data['RestingECG'] = data['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2}).values
    data['ExerciseAngina'] = data['ExerciseAngina'].map({'Y': 1, 'N': 0})
    data['ST_Slope'] = data['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})
    X = data.drop(columns=['HeartDisease'])
    y = data["HeartDisease"].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = PCA(n_components=7).fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test


algorithm_globals.random_seed = 123
X_train, X_test, y_train, y_test = data_preprocessing('heart_disease.csv')
num_features = X_train.shape[1]
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
ansatz = TwoLocal(num_qubits=num_features, rotation_blocks=['ry'], entanglement_blocks='cz', reps=5)
optimizer = COBYLA(maxiter=100)
sampler = Sampler()
vqc = VQC(sampler=sampler, feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, callback=None)
start = t.time()
print("trainign started")
vqc.fit(X_train, y_train)
end = t.time() - start
print(
    f'training score: {vqc.score(X_train, y_train)} and it took {end} seconds. Finally, the testing score was: {vqc.score(X_test, y_test)}')
