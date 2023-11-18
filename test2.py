import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC

# Step 1: Load the dataset
data = pd.read_csv('EEG_data.csv')  # Replace 'your_dataset.csv' with the actual file name

# Step 2: Data Preprocessing
X = data.iloc[:, [5,6,7,9]]  # Columns 2 to 12 are your features
#y = data['user-definedlabeln']  # Assuming the label is in the 'user-definedlabeln' column
y = data.iloc[:, 14]  # Assuming the label is in the 'user-definedlabeln' column

# Step 3: Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Choose and Initialize the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 5: Train the Model
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)


# Step 7: Now you can use the trained model to make predictions on new data
# For example, assuming new_data is a new Muse S data with the same structure as the training data
new_data = pd.read_csv('EEG_data.csv')  # Load new data
X_new = new_data.iloc[:, [5,6,7,9]].values  # Extract relevant columns
predictions = model.predict(X_new)
print("Predictions for new data:", predictions)



# Step 1: Load the dataset
data2 = pd.read_csv('EEG_data.csv')  # Replace 'your_dataset.csv' with the actual file name

# Step 2: Data Preprocessing
X2 = data2.iloc[:, [5, 6, 7, 9]]  # Columns 6, 7, 8, 10 are your features
y2 = data2.iloc[:, 14]  # Column 15 is your label

# Step 3: Data Splitting
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Step 4: Choose and Initialize the Model (SVM)
model2 = SVC(kernel='poly', random_state=42)  # You can also try 'rbf' or 'poly' kernels

# Step 5: Train the Model
model2.fit(X_train2, y_train2)

# Step 6: Evaluate the Model
y_pred2 = model2.predict(X_test2)
accuracy2 = accuracy_score(y_test2, y_pred2)
conf_matrix2 = confusion_matrix(y_test2, y_pred2)

print(f"Accuracy: {accuracy2}")
print("Confusion Matrix:")
print(conf_matrix2)
print(type(model))
print(type(model2))
