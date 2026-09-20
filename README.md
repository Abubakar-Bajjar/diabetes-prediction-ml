# Diabetes Prediction Using Random Forest Classifier

AI Lab Project — Bahria University (Artificial Intelligence and Machine Learning)

**Team:** Chaudhary Muhammad Abubakar (09-239241-028), Saad Ali (09-239241-088)
**Instructors:** Engr Ifrah, Dr Saleem

## Objective

Develop and implement a machine learning model — a Random Forest Classifier, compared against Logistic Regression — to predict the likelihood of diabetes in individuals based on key medical features, enabling early intervention and proactive healthcare management.

## Dataset

Pima Indians Diabetes dataset — 768 patient records, 8 medical features:
`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age`

Target: `Outcome` (1 = diabetic, 0 = non-diabetic)

## System Architecture

1. Collect data (768 patients)
2. Clean data (remove missing values / nulls)
3. Split data (80% train / 20% test)
4. Scale features (StandardScaler)
5. Train models (Random Forest + Logistic Regression)
6. Test models on the held-out 20%
7. Evaluate (accuracy, feature importance, confusion matrix)
8. Use the best model for real-time risk assessment via an interactive CLI

## Results

| Metric | Value |
|---|---|
| Model Accuracy (Random Forest) | 75.32% |
| Patients Analyzed | 768 |
| Medical Features | 8 |
| Most Important Feature | Glucose |

**Feature importance:** Glucose (28.45%) > BMI (19.23%) > Age (16.54%) > DiabetesPedigreeFunction (12.34%) > Pregnancies (9.87%) > Insulin (7.65%) > BloodPressure (5.43%) > SkinThickness (4.21%)

## Setup

```bash
pip install -r requirements.txt
python diabetes_prediction.py
```

The script trains both models, prints accuracy/evaluation metrics for the best one, then walks you through an interactive questionnaire (with sensible estimates for values you don't know, like insulin or skinfold thickness) and gives a diabetes risk prediction.

## Repo Structure

```
diabetes-prediction/
├── data/
│   └── diabetes.csv
├── diabetes_prediction.py
├── requirements.txt
└── README.md
```

## Disclaimer

This is an academic project for demonstrating ML classification techniques. It is not a medical diagnostic tool — always consult a healthcare professional for medical advice.
