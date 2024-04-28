#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the dataset
data = pd.read_csv("HousingData.csv")


# Step 2: Split the data into training and testing sets
X = data.drop('MEDV', axis=1)  # Features
y = data['MEDV']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Step 3: Handle missing values (if any)
# Example: fill missing values with the mean
X_train.fillna(X_train.mean(), inplace=True)


# Step 4: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)



# Step 5: Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)


# Step 6: Evaluate the model
# Preprocess X_test if necessary (e.g., fill missing values, scale features)
X_test.fillna(X_test.mean(), inplace=True)
X_test_scaled = scaler.transform(X_test)  # Assuming you used StandardScaler for feature scaling

# Predict using the model
y_pred = model.predict(X_test_scaled)

# Evaluate the predictions
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)



# Step 7: Visualize the results (optional)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()