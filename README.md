# Student Score Predictor

A machine learning project that predicts student scores based on age, AI knowledge, and ML knowledge.

The project demonstrates a beginner-friendly machine learning workflow, from exploratory data analysis and model training to model persistence and interactive prediction.

## Project Overview

This project uses a small educational dataset containing:

- Age
- AI knowledge
- ML knowledge
- Student score

The goal is to train regression models and predict the score of a new student.

## Machine Learning Workflow

''''

Dataset
↓
Exploratory Data Analysis
↓
Feature / Target Separation
↓
Train/Test Split
↓
Model Training
↓
Model Comparison
↓
Model Evaluation
↓
Model Persistence
↓
Interactive Prediction

''''

## Models & Results

Three regression models were evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

| Model             |  MAE |
| ----------------- | ---: |
| Linear Regression | 0.68 |
| Random Forest     | 2.27 |
| Decision Tree     | 4.00 |

Lower MAE indicates better performance.

Linear Regression achieved the lowest MAE on the current test split.

> **Note:** The dataset contains only 10 examples and is intended for learning purposes. These results should not be interpreted as evidence of real-world model performance.

## Features & Dataset

The model uses the following features:

| Feature | Description                        |
| ------- | ---------------------------------- |
| Age     | Student's age                      |
| AI      | AI knowledge (`0 = No`, `1 = Yes`) |
| ML      | ML knowledge (`0 = No`, `1 = Yes`) |

The target variable is `Score`.

The dataset is stored in `data/student_scores.csv`.

## Exploratory Data Analysis

The project includes analysis of:

- Dataset structure
- Missing values
- Descriptive statistics
- Age vs Score
- AI vs Score
- ML vs Score
- Feature correlations
- Correlation heatmap## Project Structure

```text
student-score-predictor/
│
├── data/
│   └── student_scores.csv
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── src/
│   ├── train.py
│   ├── compare_models.py
│   └── predict.py
│
├── model.joblib
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/piyush31-sketch/student-score-predictor.git

Move into the project directory:
cd student-score-predictor

Create a virtual environment:
python -m venv venv

Activate the virtual environment on Windows Git Bash:
source venv/Scripts/activate

Install the dependencies:
pip install -r requirements.txt

```

## Train the Model

Run:

```bash
python src/train.py

This trains the Linear Regression model and saves the trained model as model.joblib.
```

## Compare Models

Run:

```bash
python src/compare_models.py

This compares Linear Regression, Decision Tree, and Random Forest using Mean Absolute Error (MAE).
```

## Make Predictions

Run:

```bash
python src/predict.py
```

The program asks for:

- Age
- AI knowledge
- ML knowledge

Example:

```text
Enter Age: 30
Knows AI? (1 = Yes, 0 = No): 1
Knows ML? (1 = Yes, 0 = No): 1

Predicted Score: 98.59
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook
- Git & GitHub

## What I Learned

Through this project, I practiced:

- Loading and exploring datasets
- Feature and target separation
- Train/test splitting
- Linear regression
- Decision trees
- Random forests
- Model evaluation using MAE
- Exploratory data analysis
- Data visualization
- Correlation analysis
- Saving and loading trained models
- Building an interactive prediction script
- Managing Python dependencies
- Using Git and GitHub for version control

## Future Improvements

- Use a larger real-world dataset
- Add more meaningful features
- Perform cross-validation
- Tune model hyperparameters
- Add additional evaluation metrics
- Build a web interface
- Deploy the model as an API
- Add automated tests
- Add CI/CD
