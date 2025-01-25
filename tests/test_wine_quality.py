import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Assuming the dataset is in CSV format, you can adjust the file path.
def load_data():
    # Load the wine dataset (this can be adjusted based on your data source)
    df = pd.read_csv('../data/wine_quality.csv')
    return df

def test_data_loading():
    df = load_data()
    # Check if the dataset contains the expected columns
    expected_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                        'pH', 'sulphates', 'alcohol', 'quality']
    assert list(df.columns) == expected_columns, f"Expected columns: {expected_columns}, but got {list(df.columns)}"
    
    # Check that the dataset is not empty
    assert df.shape[0] > 0, "Dataset is empty"
    
    # Check the number of rows and columns
    assert df.shape == (1599, 12), f"Expected (1599, 12), but got {df.shape}"

def preprocess_data(df):
    # Example of preprocessing step: Standardization of features
    scaler = StandardScaler()
    features = df.drop('quality', axis=1)
    features_scaled = scaler.fit_transform(features)
    return features_scaled

def test_data_preprocessing():
    df = load_data()
    
    # Preprocess the data (scale the features)
    features_scaled = preprocess_data(df)
    
    # Check if features_scaled is a numpy array
    assert isinstance(features_scaled, np.ndarray), "Features should be a numpy array"
    
    # Check if the number of rows matches the original data
    assert features_scaled.shape[0] == df.shape[0], "Mismatch in number of rows after preprocessing"
    
    # Check if features have been scaled (mean should be near 0, std should be near 1)
    assert abs(features_scaled.mean()) < 0.1, "Feature means are not centered near 0"
    assert abs(features_scaled.std() - 1) < 0.1, "Feature standard deviations are not close to 1"

def train_model(df):
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def test_model_training():
    df = load_data()
    model, X_test, y_test = train_model(df)
    
    # Ensure the model is trained (check if the model has the 'feature_importances_' attribute)
    assert hasattr(model, 'feature_importances_'), "Model is not trained properly"
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Check if predictions are within a valid range (wine quality ranges from 0 to 10)
    assert all(y_pred >= 0) and all(y_pred <= 10), "Predicted values are out of range"

def test_model_predictions():
    df = load_data()
    model, X_test, y_test = train_model(df)
    
    # Predict using the trained model
    y_pred = model.predict(X_test)
    
    # Ensure predictions are numerical and within the valid range
    assert isinstance(y_pred[0], (int, float)), "Prediction is not a number"
    assert all(y_pred >= 0) and all(y_pred <= 10), "Predicted values are out of range"
    
    # Optionally, compare predictions to ground truth (just checking a few examples)
    assert abs(y_pred[0] - y_test.iloc[0]) < 5, f"Prediction error is too large: {y_pred[0]} vs {y_test.iloc[0]}"
