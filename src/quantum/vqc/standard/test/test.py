from train import vqc, X_test, y_test

print(f'testing score: {vqc.score(X_test, y_test)}')
