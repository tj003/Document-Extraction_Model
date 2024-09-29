import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load your data
df = pd.read_csv('D:\BOE\ModelCode\jsonModel\combined_preprocessed_data.csv')

# Define the target fields you want to predict
target_fields = [
    'fields.port-code', 'fields.be-no', 'fields.be-date', 'fields.be-type', 
    'fields.iecbr', 'fields.gstintype', 'fields.cb-code', 'fields.pkg', 
    'fields.gwt-kgs', 'fields.be-status', 'fields.mode', 'fields.def-be', 
    'fields.kacha', 'fields.sec-48', 'fields.reimp', 'fields.adv-be-ynp', 
    'fields.assess', 'fields.exam', 'fields.hss', 'fields.first-check', 
    'fields.prov-final', 'fields.country-of-origin', 'fields.country-of-consignment', 
    'fields.port-of-loading', 'fields.port-of-shipment', 'fields.importer-name-address', 
    'fields.ad-code', 'fields.cb-name', 'fields.aeo', 'fields.ucr', 
    'fields.bcd', 'fields.acd', 'fields.sws', 'fields.nccd', 'fields.add', 
    'fields.cvd', 'fields.igst', 'fields.gcess', 'fields.sg', 'fields.saed', 
    'fields.gsia', 'fields.tta', 'fields.health', 'fields.total-duty', 
    'fields.int', 'fields.pnlty', 'fields.fine', 'fields.totass-val', 
    'fields.tot-amount', 'fields.wbe-no', 'fields.date', 'fields.wbe-site', 
    'fields.wh-code', 'fields.submission-date', 'fields.assessment-date', 
    'fields.examination-date', 'fields.ooc-date', 'fields.submission-time', 
    'fields.assessment-time', 'fields.examination-time', 'fields.ooc-time', 
    'fields.exchange-rate', 'fields.ooc-no', 'fields.items', 'fields.language', 
    'manual_edit.fields', 'fields.item', 'fields.certificate-number', 
    'fields.typeg', 'fields.prc-level', 'fields.iec', 'fields.branch-slno']

# Separate features and targets
X = df.drop(target_fields, axis=1)  # Drop target fields from features
y = df[target_fields].copy()  # Create a copy for the target fields

# Check for missing values in features and target variable
missing_X = X.isnull().sum()
missing_y = y.isnull().sum()

print("Missing values in features:")
print(missing_X[missing_X > 0])

print("Missing values in target variable:")
print(missing_y[missing_y > 0])

# Fill NaN values for y (target fields) with mode
for col in y.columns:
    if y[col].isnull().any():
        mode_val = y[col].mode()
        if not mode_val.empty:
            y[col] = y[col].fillna(mode_val[0])  # Fill with mode if it exists
        else:
            y[col] = y[col].fillna('Unknown')  # Default value if mode is empty

# Fill NaN values for X (features) with mean or a default value
for col in X.columns:
    if X[col].isnull().any():
        if X[col].dtype == 'object':  # Categorical variable
            X[col] = X[col].fillna('Unknown')  # Fill with a default value
        else:
            X[col] = X[col].fillna(X[col].mean())  # Fill with mean for numerical variables

# Drop columns with all NaN values
X = X.dropna(axis=1, how='all')

# Convert categorical variables to numeric using OneHotEncoder for features
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables to dummy variables

# Encode target variables if they are categorical
for col in y.columns:
    if y[col].dtype == 'object':  # If the target is categorical
        le = LabelEncoder()
        y[col] = le.fit_transform(y[col])  # Convert to numeric labels

# Check again for remaining NaN values
if X.isnull().values.any():
    print("Warning: There are still NaN values in the features.")
if y.isnull().values.any():
    print("Warning: There are still NaN values in the target variable.")

# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Choose a model (RandomForestRegressor in this case)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("Validation Mean Squared Error:", mse)
print("Validation R^2 Score:", r2)

# You can also evaluate on the test set
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Test Mean Squared Error:", test_mse)
print("Test R^2 Score:", test_r2)