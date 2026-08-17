import pandas as pd

# Loading the dataset
data = pd.read_csv("data/student_scores.csv")

# Features
X = data[["Age", "AI", "ML"]]

# Target
y = data["Score"]

print("Features:")
print(X)

print("\nTarget:")
print(y)

from sklearn.model_selection import train_test_split

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nX_train:")
print(X_train)

print("\nX_test:")
print(X_test)

print("\ny_train:")
print(y_train)

print("\ny_test:")
print(y_test)

from sklearn.linear_model import LinearRegression

# Creating the model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Making predictions on the test data
predictions = model.predict(X_test)

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y_test.values)

from sklearn.metrics import mean_absolute_error

# Calculating the Mean Absolute Error
mae = mean_absolute_error(y_test, predictions)

print("\nMAE:")
print(mae)