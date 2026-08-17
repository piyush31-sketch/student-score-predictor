import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 

# Loading the dataset
data = pd.read_csv("data/student_scores.csv")

# Features
X = data[["Age", "AI", "ML"]]

# Target
y = data["Score"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

for name, model in models.items():

    # Training
    model.fit(X_train, y_train)

    # Predicting
    predictions = model.predict(X_test)

    # Calculating MAE
    mae = mean_absolute_error(y_test, predictions)

    print(f"{name}: MAE = {mae:.2f}")

