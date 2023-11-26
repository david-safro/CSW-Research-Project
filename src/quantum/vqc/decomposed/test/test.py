import joblib

def load_model_and_data():
    qsvc = joblib.load('qsvc_model.pkl')
    X_test, y_test = joblib.load('test_data.pkl')
    return qsvc, X_test, y_test

def test_model(qsvc, X_test, y_test):
    qsvc_score = qsvc.score(X_test, y_test)
    print("Accuracy on testing set: " + str(qsvc_score))

if __name__ == "__main__":
    qsvc, X_test, y_test = load_model_and_data()
    test_model(qsvc, X_test, y_test)
