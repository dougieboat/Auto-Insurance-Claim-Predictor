import pandas as pd
import numpy as np
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scaler = joblib.load('model/scaler.pkl')
        self.encoders = joblib.load('model/encoders.pkl')
        self.feature_names = joblib.load('model/feature_names.pkl')
    
    def preprocess_input(self, input_data):
        """Preprocess user input for model prediction"""
        df = pd.DataFrame([input_data])
        
        # Ensure all required columns are present
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = None
        
        # Reorder columns to match training data
        df = df[self.feature_names]
        
        # Handle categorical encoding
        categorical_columns = ['Gender', 'Car_Category', 'Car_Make', 'LGA_Name', 'State', 'Product_Name']
        
        for col in categorical_columns:
            if col in df.columns:
                # Handle unknown categories
                try:
                    df[col] = self.encoders[col].transform(df[col].astype(str))
                except ValueError:
                    # If unknown category, use the most frequent class
                    df[col] = 0
        
        # Scale numerical features
        numerical_columns = ['Age', 'Policy_Count']
        df[numerical_columns] = self.scaler.transform(df[numerical_columns])
        
        return df
    
    def get_unique_values(self):
        """Get unique values for dropdown options"""
        # Load original data to get unique values
        original_data = pd.read_csv('data/cleaned_insurance_company.csv')
        
        unique_values = {}
        categorical_columns = ['Gender', 'Car_Category', 'Car_Make', 'LGA_Name', 'State', 'Product_Name']
        
        for col in categorical_columns:
            unique_values[col] = sorted(original_data[col].unique().tolist())
        
        return unique_values