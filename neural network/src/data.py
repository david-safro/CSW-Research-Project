import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class HeartDiseasePrediction:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = None
        self.age = None
        self.sex = None
        self.chest_pain_type = None
        self.bp = None
        self.cholesterol = None
        self.fbs_over_120 = None
        self.ekg_results = None
        self.max_hr = None
        self.exercise_angina = None
        self.st_depression = None
        self.slope_of_st = None
        self.num_vessels_fluro = None
        self.thallium = None
        self.heart_disease = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def read_data(self):
        self.data = pd.read_csv(self.csv_file)

    def preprocess_data(self):
        self.age = self.data['Age']
        self.sex = self.data['Sex']
        self.chest_pain_type = self.data['Chest pain type']
        self.bp = self.data['BP']
        self.cholesterol = self.data['Cholesterol']
        self.fbs_over_120 = self.data['FBS over 120']
        self.ekg_results = self.data['EKG results']
        self.max_hr = self.data['Max HR']
        self.exercise_angina = self.data['Exercise angina']
        self.st_depression = self.data['ST depression']
        self.slope_of_st = self.data['Slope of ST']
        self.num_vessels_fluro = self.data['Number of vessels fluro']
        self.thallium = self.data['Thallium']
        self.heart_disease = self.data['Heart Disease']

        self.X = self.data.drop('Heart Disease', axis=1)
        self.y = self.data['Heart Disease']

        scaler = MinMaxScaler()
        self.X_scaled = scaler.fit_transform(self.X)

    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=random_state
        )

    def get_row_by_index(self, index):
        return list(self.data.iloc[index])

    def get_column_data(self, column_name):
        return list(self.data[column_name])

    def print_sample_data(self):
        print("Sample data:")
        print(self.data.head())

    def print_scaled_features(self):
        print("Scaled features:")
        print(self.X_scaled[:5])

    def print_data_shapes(self):
        print("Training set shape:", self.X_train.shape, self.y_train.shape)
        print("Testing set shape:", self.X_test.shape, self.y_test.shape)
