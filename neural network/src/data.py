import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def read_and_preprocess_data(csv_file):
    data = pd.read_csv(csv_file)
    X = data.drop(columns=['Heart Disease'])
    y = data['Heart Disease'].map({'Presence': 1, 'Absence': 0}).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler
