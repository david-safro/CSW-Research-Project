import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    return X_train, X_test,  y_train, y_test, scaler