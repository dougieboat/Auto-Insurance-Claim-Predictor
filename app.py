import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from utils.preprocessing import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')

# Page setup
st.set_page_config(
    page_title="Insurance Claims Predictor",
    page_icon="🚗",
    layout="wide"
)

# Load model function
@st.cache_resource
def load_model_components():
    """Load the trained model and preprocessors"""
    try:
        model = joblib.load('model/trained_model.pkl')
        preprocessor = DataPreprocessor()
        return model, preprocessor
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run train_model.py first.")
        return None, None

def main():
    # Title
    st.title("🚗 Insurance Claims Predictor")
    st.markdown("### Predict if a customer will make a claim in the first 3 months")
    
    # Load model
    model, preprocessor = load_model_components()
    
    if model is None or preprocessor is None:
        st.stop()
    
    # Get options for dropdowns
    unique_values = preprocessor.get_unique_values()
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📝 Customer Information")
        
        # Input fields
        gender = st.selectbox("Gender", unique_values['Gender'])
        age = st.slider("Age", min_value=18, max_value=80, value=40)
        policy_count = st.selectbox("Number of Policies", options=[1, 2, 3, 4, 5])
        car_category = st.selectbox("Car Category", unique_values['Car_Category'])
        
    with col2:
        st.header("🚗 Vehicle & Location Info")
        
        car_make = st.selectbox("Car Make", unique_values['Car_Make'])
        lga_name = st.selectbox("Local Government Area", unique_values['LGA_Name'])
        state = st.selectbox("State", unique_values['State'])
        product_name = st.selectbox("Insurance Product", unique_values['Product_Name'])
    
    # Predict button
    if st.button("🔮 Predict Claim Risk", type="primary"):
        # Prepare input data
        input_data = {
            'Gender': gender,
            'Age': age,
            'Policy_Count': policy_count,
            'Car_Category': car_category,
            'Car_Make': car_make,
            'LGA_Name': lga_name,
            'State': state,
            'Product_Name': product_name
        }
        
        try:
            # Make prediction
            processed_data = preprocessor.preprocess_input(input_data)
            prediction = model.predict(processed_data)[0]
            prediction_proba = model.predict_proba(processed_data)[0]
            
            # Show results
            st.header("🎯 Prediction Results")
            
            if prediction == 1:
                st.error(f"⚠️ HIGH RISK: Customer likely to make a claim (Probability: {prediction_proba[1]:.2%})")
            else:
                st.success(f"✅ LOW RISK: Customer unlikely to make a claim (Probability: {prediction_proba[0]:.2%})")
            
            # Show probability chart
            fig = go.Figure(go.Bar(
                x=['No Claim', 'Will Claim'],
                y=[prediction_proba[0], prediction_proba[1]],
                marker_color=['green', 'red'],
                text=[f'{prediction_proba[0]:.2%}', f'{prediction_proba[1]:.2%}'],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Prediction Confidence",
                yaxis_title="Probability",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()