import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib  # For saving the model

def main():
    data = pd.read_csv('data.csv')

    X = data.drop('HeartDisease', axis=1)
    y = data['HeartDisease']

    numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(drop='first'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(random_state=42))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)

    # Saving the model and the test set
    joblib.dump(pipeline, 'model.pkl')
    joblib.dump((X_test, y_test), 'test_data.pkl')

if __name__ == "__main__":
    main()
