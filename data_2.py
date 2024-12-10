import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib


# Load and prepare data
data = pd.read_sas('C:/Users/PRAVEENA/Downloads/capstone_project/P_DR2IFF.xpt')

# Select only numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns
data = data[numeric_columns]

target_column = 'DR2IP226'

# Handle missing values
data_clean = data.dropna(subset=[target_column])
data_clean = data_clean.fillna(data_clean.median())

# Separate features and target
X = data_clean.drop(target_column, axis=1)
y = data_clean[target_column]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best hyperparameters
print("Best parameters found: ", grid_search.best_params_)

# Use the best model to predict and evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Print evaluation metrics
print(f"\nModel Performance with Best Parameters:")
print(f"R2 Score: {r2_score(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"Root Mean Squared Error: {mean_squared_error(y_test, y_pred, squared=False)}")

# Print feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
})
print("\nTop 10 Most Important Features:")
print(feature_importance.sort_values('importance', ascending=False).head(10))


# Save the best model
joblib.dump(best_model, 'personalized_healthcare_model.pkl')