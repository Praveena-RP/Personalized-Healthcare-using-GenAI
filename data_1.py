import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_sas('C:/Users/PRAVEENA/Downloads/capstone_project/DR2IFF_L.xpt')

# Select only numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns
data = data[numeric_columns]

# Replace 'DR2IP227' with the correct column name (e.g., 'DR2IP225')
target_column = 'DR2IP226'  # or the correct column name

# Handle missing values by dropping rows with missing target values
data_clean = data.dropna(subset=[target_column])

# Fill remaining missing values with median
data_clean = data_clean.fillna(data_clean.median())

# Separate features and target from cleaned data
X = data_clean.drop(target_column, axis=1)  # Features
y = data_clean[target_column]  # Target

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the model with fewer trees for faster processing
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(f"R2 Score: {r2_score(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"Root Mean Squared Error: {mean_squared_error(y_test, y_pred, squared=False)}")

# Print feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
print("\nTop 10 Most Important Features:")
print(feature_importance.sort_values('importance', ascending=False).head(10))
