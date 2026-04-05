import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. GENERATE SYNTHETIC DATA (Representative of real-world loan data)
# Note: In a real project, you would use: df = pd.read_csv('loan_data.csv')
def create_sample_data(n=200):
    np.random.seed(42)
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Married': np.random.choice(['Yes', 'No'], n),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n),
        'ApplicantIncome': np.random.randint(2000, 10000, n),
        'LoanAmount': np.random.randint(100, 500, n),
        'Credit_History': np.random.choice([1.0, 0.0], n, p=[0.75, 0.25]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n),
        'Loan_Status': []
    }
    
    # Logic for Loan_Status (Higher chance if Credit_History = 1 and Income > Amount)
    for i in range(n):
        score = 0
        if data['Credit_History'][i] == 1.0: score += 50
        if data['ApplicantIncome'][i] > data['LoanAmount'][i] * 10: score += 30
        if data['Education'][i] == 'Graduate': score += 10
        
        status = 'Y' if score >= 60 else 'N'
        data['Loan_Status'].append(status)
        
    return pd.DataFrame(data)

# 2. DATA PREPROCESSING
df = create_sample_data(500)

# Encoding categorical variables
le = LabelEncoder()
categorical_cols = ['Gender', 'Married', 'Education', 'Property_Area', 'Loan_Status']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Features and Target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# 3. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. MODEL TRAINING (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. EVALUATION
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. SAMPLE PREDICTION (Testing a new applicant)
# [Gender, Married, Education, ApplicantIncome, LoanAmount, Credit_History, Property_Area]
# Example: Male (1), Yes (1), Graduate (0), 5000, 150, 1.0, Urban (2)
new_applicant = np.array([[1, 1, 0, 5000, 150, 1.0, 2]])
prediction = model.predict(new_applicant)
print("\n--- SAMPLE PREDICTION ---")
print(f"Status for New Applicant: {'APPROVED (Y)' if prediction[0] == 1 else 'REJECTED (N)'}")

print("\n--- Project Ready for Use ---")
