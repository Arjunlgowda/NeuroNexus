# Titanic Survival Prediction 🚢

This project is part of the NeuroNexus Internship task. The goal is to build a machine learning model that predicts whether a passenger survived the Titanic disaster based on features like age, sex, class, fare, etc.

## 📌 Problem Statement

Using the Titanic dataset, create a predictive model that determines whether a passenger survived or not.

## 📁 Dataset

- The dataset is available [here](https://www.kaggle.com/competitions/titanic/data).
- Key features:
  - `Pclass`: Ticket class
  - `Sex`: Gender
  - `Age`: Age of the passenger
  - `SibSp`: Number of siblings/spouses aboard
  - `Parch`: Number of parents/children aboard
  - `Fare`: Ticket fare
  - `Embarked`: Port of embarkation
  - `Survived`: Target variable (0 = No, 1 = Yes)

## 🧪 Technologies Used

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- scikit-learn (Logistic Regression)
- streamlit

## 🔍 Steps Followed

1. **Data Exploration (EDA):**
   - Analyzed dataset for patterns and missing values
   - Visualized survival distribution based on features

2. **Data Preprocessing:**
   - Handled missing values (`Age`, `Embarked`)
   - Converted categorical variables to numerical
   - Dropped irrelevant columns (`Name`, `Ticket`, `Cabin`)

3. **Model Building:**
   - Used Logistic Regression as the classification model
   - Split dataset into training and testing sets

4. **Evaluation:**
   - Evaluated using accuracy, confusion matrix, and classification report

## 📊 Results

- Model Accuracy: **~90%** (may vary depending on preprocessing)
- Metrics: Precision, Recall, F1-score


## 🏁 How to Run

1. Clone the repository
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   python titanic_model_pipeline.py
   streamlit run app.py

