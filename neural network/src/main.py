from data import HeartDiseasePrediction

# Usage example:
csv_file = 'Heart_Disease_Prediction.csv'
prediction = HeartDiseasePrediction(csv_file)
prediction.read_data()
prediction.preprocess_data()
prediction.split_data()

# Access the attributes or call other methods as needed
print("Sample data:")
print(prediction.data.head())

print("Scaled features:")
print(prediction.X_scaled[:5])

print("Training set shape:", prediction.X_train.shape, prediction.y_train.shape)
print("Testing set shape:", prediction.X_test.shape, prediction.y_test.shape)
