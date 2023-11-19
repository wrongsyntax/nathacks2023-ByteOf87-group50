import os
import time
import pandas as pd
#from sklearn.externals import joblib
from joblib import load
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = load('random_forest_model.joblib')

'''
scaler = StandardScaler()
scaler.fit(model.fit_transform(model.predict(scaler.transform(X))))
'''

# Start the neurofeedback.py process
os.system('python3 neurofeedback.py')

# Read the csv file and predict
while True:
    try:
        # Sleep for a few seconds to wait for the csv file to be written
        time.sleep(3)

        # Read the csv file
        df = pd.read_csv('live_data.csv')

        # Check if the csv file is empty
        if df.empty:
            continue

        # Scale the data
#        data = scaler.transform(df.values)

        # Predict using the saved model
        predictions = model.predict(df)
        print(predictions)

        # Delete the rows from the csv file that have been read
        with open('live_data.csv', 'w') as f:
            f.truncate(0)

    except KeyboardInterrupt:
        print('Closing!')