import sys
import os

# Add the 'models' directory to the Python path
models_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'models')

# Add it to the Python path
sys.path.append(models_path)
print(os.path.abspath(os.path.dirname(__file__)), '..')

# Import necessary functions from models.py
from models import load_data, preprocess_data, train_model, save_model, load_saved_model, predict

def main():
    # Step 1: Load the data
    print("Loading data...")
    df = load_data('../data/wine_quality.csv')  # Adjust the path if necessary
    
    # Step 2: Preprocess the data
    print("Preprocessing data...")
    X, y = preprocess_data(df)
    
    # Step 3: Train the model
    print("Training model...")
    model = train_model(X, y)
    
    # Step 4: Save the trained model
    print("Saving model...")
    save_model(model)
    
    # Step 5: Load the model (in case you want to load it for prediction)
    print("Loading the saved model...")
    loaded_model = load_saved_model('../models/wine_quality_model.pkl')
    
    # Step 6: Make predictions with the trained model
    print("Making predictions...")
    predictions = predict(loaded_model, X)
    
    # Print the first 10 predictions to verify
    print("First 10 predictions:", predictions[:10])

if __name__ == "__main__":
    main()