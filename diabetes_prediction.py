"""
Diabetes Prediction System
AI Lab Project - Bahria University

Trains and compares two algorithms (Random Forest + Logistic Regression)
on the Pima Indians Diabetes dataset, then runs an interactive
risk-assessment tool using the better-performing model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DIABETES PREDICTION SYSTEM")
print("=" * 70)

# ================================================================================
# STEP 1: LOAD AND PREPARE DATA
# ================================================================================
print("\n[1] Loading Dataset...")
df = pd.read_csv('data/diabetes.csv')
df.columns = df.columns.str.strip()  # fixes the stray space in "SkinThickness "

print(f"Dataset loaded: {len(df)} patients, {len(df.columns) - 1} features\n")

print("[2] Preparing Data...")
df = df.dropna()
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================================================================
# STEP 2: TRAIN + COMPARE TWO ALGORITHMS
# ================================================================================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

results = {}
print("[3] Training and Comparing Models...\n")
for name, clf in models.items():
    clf.fit(X_train_scaled, y_train)
    preds = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = {"model": clf, "accuracy": acc, "preds": preds}
    print(f"  {name}: {acc * 100:.2f}% accuracy")

best_name = max(results, key=lambda n: results[n]["accuracy"])
model = results[best_name]["model"]
print(f"\n[4] Best model: {best_name} ({results[best_name]['accuracy'] * 100:.2f}%)")

print("\n" + "=" * 70)
print(f"MODEL EVALUATION — {best_name}")
print("=" * 70)
print(classification_report(y_test, results[best_name]["preds"],
                             target_names=['No Diabetes', 'Has Diabetes']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, results[best_name]["preds"]))

if hasattr(model, "feature_importances_"):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nTop Features:")
    print(feature_importance.head())

# ================================================================================
# STEP 3: INTERACTIVE PREDICTION
# ================================================================================

def calculate_bmi(weight_kg, height_input, height_unit):
    """Calculate BMI from weight and height"""
    if height_unit.lower() == 'cm':
        height_m = height_input / 100
    else:  # feet.inches format, e.g. 5.8 -> 5'8"
        feet = int(height_input)
        inches = (height_input - feet) * 10
        height_m = (feet * 12 + inches) * 0.0254
    return round(weight_kg / (height_m ** 2), 2)


def get_yes_no_input(question):
    while True:
        response = input(question + " (Yes/No): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        if response in ['no', 'n']:
            return False
        print("Please enter Yes or No")


def get_number_input(question, min_val=0, max_val=None):
    while True:
        try:
            value = float(input(question))
            if value < min_val:
                print(f"Value must be at least {min_val}")
                continue
            if max_val and value > max_val:
                print(f"Value must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("Please enter a valid number")


def predict_diabetes_enhanced():
    print("\n" + "=" * 70)
    print("DIABETES RISK ASSESSMENT - PATIENT INFORMATION")
    print("=" * 70)

    try:
        print("\n--- BASIC INFORMATION ---")
        age = get_number_input("Enter your age (years): ", 1, 120)

        gender = input("Gender (Male/Female): ").strip().lower()
        if gender in ['female', 'f']:
            pregnancies = get_number_input("Number of pregnancies: ", 0, 20)
        else:
            pregnancies = 0
            print("Pregnancies set to 0 (Male)")

        print("\n--- BMI CALCULATOR ---")
        weight = get_number_input("Enter your weight (kg): ", 20, 300)
        height_unit = input("Height unit (cm/feet): ").strip().lower()
        if height_unit == 'cm':
            height = get_number_input("Enter height (cm): ", 50, 250)
        else:
            print("Enter height in feet.inches format (e.g., 5.8 for 5 feet 8 inches)")
            height = get_number_input("Enter height: ", 3, 8)
        bmi = calculate_bmi(weight, height, height_unit)
        print(f"Calculated BMI: {bmi}")

        print("\n--- BLOOD PRESSURE ---")
        if get_yes_no_input("Do you know your blood pressure reading?"):
            blood_pressure = get_number_input("Enter systolic BP (e.g., 120): ", 60, 200)
        else:
            blood_pressure = 90 if get_yes_no_input("Do you have high blood pressure?") else 72
            print(f"Estimated BP: {blood_pressure}")

        print("\n--- BLOOD SUGAR ---")
        if get_yes_no_input("Have you checked blood sugar recently?"):
            glucose = get_number_input("Enter glucose level (mg/dL): ", 50, 300)
        else:
            print("Answer these questions to estimate glucose:")
            feel_thirsty = get_yes_no_input("  Do you feel very thirsty often?")
            frequent_urination = get_yes_no_input("  Do you urinate frequently?")
            feel_tired = get_yes_no_input("  Do you feel tired/weak often?")
            glucose = 100
            if feel_thirsty: glucose += 25
            if frequent_urination: glucose += 20
            if feel_tired: glucose += 15
            print(f"Estimated glucose: {glucose} mg/dL")

        print("\n--- INSULIN LEVEL ---")
        if get_yes_no_input("Do you know your insulin level?"):
            insulin = get_number_input("Enter insulin level (uU/mL): ", 0, 500)
        else:
            if bmi > 30 and glucose > 140:
                insulin = 150
            elif bmi > 25 or glucose > 120:
                insulin = 100
            else:
                insulin = 80
            print(f"Estimated insulin: {insulin} uU/mL")

        print("\n--- SKIN THICKNESS ---")
        if get_yes_no_input("Do you know your triceps skinfold thickness?"):
            skin_thickness = get_number_input("Enter thickness (mm): ", 0, 100)
        else:
            skin_thickness = 35 if bmi > 30 else (25 if bmi > 25 else 20)
            print(f"Estimated skin thickness: {skin_thickness} mm")

        print("\n--- FAMILY HISTORY ---")
        if get_yes_no_input("Does anyone in your family have diabetes?"):
            diabetes_pedigree = 0.8 if get_yes_no_input("  Is it a close relative (parent/sibling)?") else 0.4
        else:
            diabetes_pedigree = 0.2
        print(f"Diabetes pedigree function: {diabetes_pedigree}")

        print("\n" + "=" * 70)
        print("ANALYZING PATIENT DATA...")
        print("=" * 70)

        new_patient = np.array([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree, age
        ]])

        print("\nPatient Summary:")
        print(f"  Age: {age} years")
        print(f"  Pregnancies: {pregnancies}")
        print(f"  BMI: {bmi}")
        print(f"  Blood Pressure: {blood_pressure} mmHg")
        print(f"  Glucose: {glucose} mg/dL")
        print(f"  Insulin: {insulin} uU/mL")
        print(f"  Skin Thickness: {skin_thickness} mm")
        print(f"  Diabetes Pedigree: {diabetes_pedigree}")

        new_patient_scaled = scaler.transform(new_patient)
        prediction = model.predict(new_patient_scaled)[0]
        probability = model.predict_proba(new_patient_scaled)[0]

        print("\n" + "=" * 70)
        print(f"PREDICTION RESULT (model: {best_name})")
        print("=" * 70)

        if prediction == 1:
            print("WARNING: HIGH RISK OF DIABETES")
            print(f"   Confidence: {probability[1] * 100:.1f}%")
            print("\n   Recommendations:")
            print("   - Consult a doctor immediately")
            print("   - Get blood sugar tested")
            print("   - Monitor diet and exercise")
        else:
            print("LOW RISK OF DIABETES")
            print(f"   Confidence: {probability[0] * 100:.1f}%")
            print("\n   Recommendations:")
            print("   - Maintain healthy lifestyle")
            print("   - Regular check-ups")
            print("   - Stay active and eat balanced diet")

        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nPrediction cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    predict_diabetes_enhanced()
    print("\nDiabetes Prediction System Completed!")
