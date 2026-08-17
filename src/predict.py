import joblib
import pandas as pd

# Loading the trained model
model = joblib.load("model.joblib")

print("Model loaded successfully!")

# Getting user input
age = float(input("Enter Age: "))
ai = int(input("Knows AI? (1 = Yes, 0 = No): "))
ml = int(input("Knows ML? (1 = Yes, 0 = No): "))

# Creating input DataFrame
new_student = pd.DataFrame({
    "Age": [age],
    "AI": [ai],
    "ML": [ml]
})

# Making prediction
prediction = model.predict(new_student)

print(f"\nPredicted Score: {prediction[0]:.2f}")