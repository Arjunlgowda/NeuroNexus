import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline# type: ignore
from sklearn.impute import SimpleImputer# type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler# type: ignore
from sklearn.ensemble import RandomForestClassifier# type: ignore
from sklearn.model_selection import train_test_split# type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score# type: ignore
import joblib# type: ignore
import matplotlib.pyplot as plt# type: ignore
import seaborn as sns# type: ignore

def load_data(path="tested.csv"):
    df = pd.read_csv(path)
    df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
    return df

def build_pipeline():
    # Features
    numerical_features = ["Age", "Fare", "SibSp", "Parch"]
    categorical_features = ["Sex", "Embarked", "Pclass"]

    # Numerical pipeline
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Full preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # Complete pipeline with model
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    return pipeline

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("\n🎯 Accuracy:", accuracy_score(y_test, predictions))
    print("\n📋 Classification Report:\n", classification_report(y_test, predictions))
    print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, predictions))

def main():
    # Load and split data
    df = load_data("tested.csv")
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build pipeline and train
    model_pipeline = build_pipeline()
    model_pipeline.fit(X_train, y_train)

    # Evaluate
    evaluate_model(model_pipeline, X_test, y_test)

    # Save model
    joblib.dump(model_pipeline, "titanic_model_pipeline.pkl")
    print("\n💾 Model saved as 'titanic_model_pipeline.pkl'")

if __name__ == "__main__":
    main()
