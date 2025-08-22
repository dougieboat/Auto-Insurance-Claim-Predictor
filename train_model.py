import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.combine import SMOTEENN
import joblib
import os

def train_and_save_model():
    print("Loading data...")
    # Load your cleaned data
    df = pd.read_csv('data/cleaned_insurance_company.csv')
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Feature selection
    feature_columns = ['Gender', 'Age', 'Policy_Count', 'Car_Category', 
                      'Car_Make', 'LGA_Name', 'State', 'Product_Name']
    target_column = 'Target'
    
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    print("Preprocessing data...")
    # Initialize encoders
    label_encoders = {}
    categorical_columns = ['Gender', 'Car_Category', 'Car_Make', 'LGA_Name', 'State', 'Product_Name']
    
    # Encode categorical variables
    for col in categorical_columns:
        print(f"Encoding {col}...")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_columns = ['Age', 'Policy_Count']
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    
    print("Splitting data...")
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Applying SMOTEENN...")
    # Apply SMOTEENN
    smoteenn = SMOTEENN(random_state=42)
    X_train_balanced, y_train_balanced = smoteenn.fit_resample(X_train, y_train)
    
    print("Training model...")
    # Train Gradient Boosting model
    gb_model = GradientBoostingClassifier(random_state=42, n_estimators=100)
    gb_model.fit(X_train_balanced, y_train_balanced)
    
    print("Evaluating model...")
    # Evaluate model
    y_pred = gb_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Saving model...")
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Save model and preprocessors
    joblib.dump(gb_model, 'model/trained_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(label_encoders, 'model/encoders.pkl')
    joblib.dump(feature_columns, 'model/feature_names.pkl')
    
    print("✅ Model and preprocessors saved successfully!")
    return gb_model, scaler, label_encoders

if __name__ == "__main__":
    train_and_save_model()