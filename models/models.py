import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os, sys

# Function to load data
def load_data(filepath='data/winequality-red.csv'):
    """
    Load the Wine Quality dataset from a CSV file.
    :param filepath: Path to the CSV file.
    :return: Pandas DataFrame with the wine dataset.
    """
    df = pd.read_csv(filepath)
    return df

# Function for preprocessing the data
def preprocess_data(df):
    """
    Preprocess the Wine Quality dataset by separating features and labels,
    and scaling the features.
    :param df: DataFrame containing the raw dataset.
    :return: Scaled features as a NumPy array, target labels as a Pandas Series.
    """
    # Separate the features (X) and the target (y)
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Function to train the model
def train_model(X, y):
    """
    Train a RandomForestRegressor model on the given data.
    :param X: Features (preprocessed data).
    :param y: Target labels.
    :return: Trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the model (RandomForestRegressor)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error on Test Set: {mae}')
    
    return model

# Function to make predictions using the trained model
def predict(model, X):
    """
    Use a trained model to make predictions on new data.
    :param model: Trained machine learning model.
    :param X: Features (preprocessed data).
    :return: Predictions.
    """
    predictions = model.predict(X)
    return predictions

# Function to save the trained model
def save_model(model):
    """
    Save the trained model to a file.
    :param model: The trained model to save.
    :param filename: The name of the file to save the model as.
    """
    model_save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'models', 'wine_quality_model.pkl')

# Example: Save model using joblib (if you're using joblib)
    joblib.dump(model, model_save_path)
    #joblib.dump(model, filename)
    print(f'Model saved as {model_save_path}')

# Function to load a saved model
def load_saved_model(filename='../models/wine_quality_model.pkl'):
    """
    Load a trained model from a file.
    :param filename: The name of the file to load the model from.
    :return: The loaded model.
    """
    model = joblib.load(filename)
    print(f'Model loaded from {filename}')
    return model
