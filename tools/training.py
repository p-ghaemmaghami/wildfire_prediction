#Importing Libraries
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, confusion_matrix

import joblib

def size_ha_classifier(size):
  if size > 0.1:
    return 'yes'
  else:
    return 'no'
  
# Loading the dataset
file_path = './data/NFDB_point_txt/NFDB_point_20240613.txt' 
df = pd.read_csv(file_path)

# Feature Engineering and Feature Selection
df['SIZE_HA_CLASS'] = df['SIZE_HA'].apply(size_ha_classifier)
print(df[['SIZE_HA', 'SIZE_HA_CLASS']].head(10))

# Select relevant features and target variable
features = ['LATITUDE', 'LONGITUDE', 'YEAR', 'MONTH', 'DAY']  
target = 'SIZE_HA_CLASS'

# Handle missing values
df = df.dropna(subset=[target] + features)  # Remove rows with NaN in target or features

# Split data into training and testing sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [2, 5, 10],
    # 'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4]
}

# Create the model
rf_classifier = RandomForestClassifier(random_state=42, verbose=True)

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1) 
grid_search.fit(X_train, y_train)

# Get the best model
model = grid_search.best_estimator_
print(f"Best hyperparameters: {grid_search.best_params_}")
# Train the model on the whole dataset
model.fit(X, y)

# Save the model to a file
joblib.dump(model, 'tools/model.joblib')