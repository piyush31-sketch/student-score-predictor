import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#loading the dataset
data = pd.read_csv("data/student_scores.csv")

#feartures and targets
X = data[["Age", "AI", "ML"]]
y = data["Score"]

# spliting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# creating and training the model
model = LinearRegression()
model.fit(X_train, y_train)

#getting user input
age = float(input("Enter Age: "))
ai = int(input("Knows AI? (1 = Yes, 0 = No): "))
ml = int(input("Knows ML? (1 = Yes, 0 = No): "))

# creating a dataframe for the input
input_data = pd.DataFrame({
    "Age": [age],
    "AI": [ai],
    "ML": [ml]
})

# making predictions
prediction = model.predict(input_data)

print(f"\nPredicted Score: {prediction[0]:.2f}")

