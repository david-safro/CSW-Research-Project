import joblib
from sklearn.metrics import accuracy_score

def load_model_and_data():
    model = joblib.load('model.pkl')
    X_test, y_test = joblib.load('test_data.pkl')
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    model, X_test, y_test = load_model_and_data()
    evaluate_model(model, X_test, y_test)
