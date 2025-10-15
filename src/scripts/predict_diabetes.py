#!/usr/bin/env python3
"""
Diabetes Prediction Script

This script demonstrates how to use the trained models to make predictions
on new data. It loads a saved model and makes predictions on sample data.

Usage:
    python src/scripts/predict_diabetes.py --model decision_tree
    python src/scripts/predict_diabetes.py --model logistic_regression --sample-data
"""

import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from core.config_loader import config_loader, MODELS_DIR, DATASET_FILE


def load_model(model_name: str):
    """Load a trained model."""
    model_path = MODELS_DIR / f"{model_name}_classifier.pkl"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Available models:")
        for pkl_file in MODELS_DIR.glob("*.pkl"):
            print(f"   - {pkl_file.stem}")
        return None
    
    try:
        model = joblib.load(model_path)
        print(f"✅ Loaded model: {model_name}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def get_sample_data():
    """Get sample data for prediction."""
    # Load original dataset to get feature names and sample data
    df = pd.read_csv(DATASET_FILE)
    X = df.drop('Diabetes_binary', axis=1)
    
    # Get a few samples
    sample_data = X.sample(n=5, random_state=42)
    return sample_data, X.columns.tolist()


def create_custom_sample():
    """Create a custom sample with typical values."""
    # Feature order based on the dataset
    features = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]
    
    # Example: 45-year-old person with some risk factors
    sample_person = {
        'HighBP': 1,           # Has high blood pressure
        'HighChol': 1,         # Has high cholesterol  
        'CholCheck': 1,        # Had cholesterol check
        'BMI': 28,             # BMI of 28 (overweight)
        'Smoker': 0,           # Non-smoker
        'Stroke': 0,           # No stroke history
        'HeartDiseaseorAttack': 0,  # No heart disease
        'PhysActivity': 1,     # Physically active
        'Fruits': 1,           # Eats fruits
        'Veggies': 1,          # Eats vegetables
        'HvyAlcoholConsump': 0, # No heavy alcohol consumption
        'AnyHealthcare': 1,    # Has healthcare coverage
        'NoDocbcCost': 0,      # Can afford doctor
        'GenHlth': 3,          # Good general health (1-5 scale)
        'MentHlth': 0,         # No mental health issues
        'PhysHlth': 2,         # Some physical health issues
        'DiffWalk': 0,         # No difficulty walking
        'Sex': 1,              # Male
        'Age': 8,              # Age group 8 (45-49 years)
        'Education': 5,        # College graduate
        'Income': 6            # Good income level
    }
    
    return pd.DataFrame([sample_person]), features


def make_predictions(model, data: pd.DataFrame, feature_names: list):
    """Make predictions using the loaded model."""
    try:
        # Make predictions
        predictions = model.predict(data)
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(data)
        elif hasattr(model, 'decision_function'):
            # For SVM
            decision_scores = model.decision_function(data)
            # Convert to probabilities (approximate)
            probabilities = np.column_stack([1 - decision_scores, decision_scores])
        else:
            probabilities = None
        
        return predictions, probabilities
    
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return None, None


def display_results(data: pd.DataFrame, predictions: np.ndarray, probabilities: np.ndarray, feature_names: list):
    """Display prediction results in a user-friendly format."""
    print("\n" + "="*60)
    print("🔮 DIABETES PREDICTION RESULTS")
    print("="*60)
    
    for i in range(len(predictions)):
        print(f"\n👤 Person {i+1}:")
        print("-" * 30)
        
        # Show prediction
        diabetes_risk = "HIGH RISK" if predictions[i] == 1 else "LOW RISK"
        risk_color = "🔴" if predictions[i] == 1 else "🟢"
        print(f"{risk_color} Diabetes Risk: {diabetes_risk}")
        
        # Show probability if available
        if probabilities is not None:
            if probabilities.ndim > 1:
                prob_diabetes = probabilities[i, 1] * 100
            else:
                prob_diabetes = probabilities[i] * 100
            print(f"📊 Probability of Diabetes: {prob_diabetes:.1f}%")
        
        # Show key risk factors
        print("📋 Key Health Indicators:")
        person_data = data.iloc[i]
        
        risk_factors = []
        if person_data['HighBP'] == 1:
            risk_factors.append("High Blood Pressure")
        if person_data['HighChol'] == 1:
            risk_factors.append("High Cholesterol")
        if person_data['BMI'] >= 25:
            risk_factors.append(f"BMI: {person_data['BMI']:.1f} (Overweight)")
        if person_data['Smoker'] == 1:
            risk_factors.append("Smoker")
        if person_data['PhysActivity'] == 0:
            risk_factors.append("Low Physical Activity")
        
        if risk_factors:
            for factor in risk_factors:
                print(f"   ⚠️  {factor}")
        else:
            print("   ✅ No major risk factors detected")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Make diabetes predictions using trained models')
    parser.add_argument('--model', required=True, 
                       choices=['decision_tree', 'logistic_regression', 'random_forest', 'svm'],
                       help='Model to use for prediction')
    parser.add_argument('--sample-data', action='store_true',
                       help='Use sample data from the original dataset')
    parser.add_argument('--custom', action='store_true',
                       help='Use custom sample person data')
    
    args = parser.parse_args()
    
    print("🧠 Diabetes Prediction Tool")
    print("=" * 40)
    
    # Load model
    model = load_model(args.model)
    if model is None:
        sys.exit(1)
    
    # Get data for prediction
    if args.sample_data:
        print("📊 Using sample data from original dataset...")
        data, feature_names = get_sample_data()
    else:
        print("👤 Using custom sample person...")
        data, feature_names = create_custom_sample()
    
    # Make predictions
    print(f"🔮 Making predictions with {args.model.replace('_', ' ').title()} model...")
    predictions, probabilities = make_predictions(model, data, feature_names)
    
    if predictions is None:
        sys.exit(1)
    
    # Display results
    display_results(data, predictions, probabilities, feature_names)
    
    print("\n" + "="*60)
    print("ℹ️  Note: These predictions are for educational purposes only.")
    print("   Always consult healthcare professionals for medical advice.")
    print("="*60)


if __name__ == "__main__":
    main()