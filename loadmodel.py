import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import load
import time, os

model = load('random_forest_model.joblib')  # Update with the actual file path where your trained model is saved

# Start the neurofeedback.py process
os.system('python3 neurofeedback.py')

while True:
    time.sleep(2)
    # Step 2: Load and preprocess live data
    live_data = pd.read_csv('live_data.csv')  # Replace 'your_live_data.csv' with the actual file name
    live_X = live_data.iloc[:, [0, 1, 2, 3]]  # Assuming columns 0, 1, 2, 3 are your features
    # Perform any necessary preprocessing on live_X, like handling missing values, normalization, etc.

    # Step 3: Make Predictions
    predictions = model.predict(live_X)

    # Print or use predictions as needed
    print("Predictions for live data:", predictions)
    with open('live_data.csv', 'w') as f:
        f.truncate(22)
