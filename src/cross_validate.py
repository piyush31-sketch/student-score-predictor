import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

# Loading dataset
data = pd.read_csv("data/student_scores.csv")

# Features and target
X = data[["Age", "AI", "ML"]]
y = data["Score"]

# Creating model
model = LinearRegression()

# Create 5-fold cross-validation
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Calculate MAE using cross-validation
scores = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="neg_mean_absolute_error"
)

# Convert negative MAE values to positive
mae_scores = -scores

print("Cross-validation MAE scores:")
print(mae_scores)

print("\nAverage CV MAE:")
print(mae_scores.mean())