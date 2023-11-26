from train import main
from sklearn.metrics import accuracy_score

def evaluate_model(classifier, X_test_transformed, y_test):
    print("Making predictions with trained QNN...")
    y_pred = classifier.predict(X_test_transformed)
    print("Predictions made successfully!")

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of the QNN is: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    classifier, X_test_transformed, y_test = main()
    evaluate_model(classifier, X_test_transformed, y_test)
