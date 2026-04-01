import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Import the original, untouched pipeline functions
from ddos_pipeline import load_and_sample_data, preprocess_data

def build_and_save_models():
    print("===========================================")
    print("Bridge Script: Training & Exporting Models ")
    print("===========================================")
    
    data_dir = 'd:/final_aics/train/'
    models_dir = 'd:/final_aics/models/'
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load original data using the untouched parser
    print("-> Loading and sampling data natively...")
    df = load_and_sample_data(data_dir, samples_per_file=2000)
    
    # 2. Leverage original preprocessing
    print("-> Preprocessing and removing zero-variance/high-null columns...")
    X, y = preprocess_data(df)
    
    # Save the exact expected column names for the Streamlit inference engine
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.joblib'))
    
    # 3. Train-test split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Scale using robust scaler
    print("-> Scaling training metrics...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    
    # 5. Over-sample Imbalanced target arrays
    print("-> Balancing class distribution via SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # 6. Fit and Persist Random Forest
    print("-> Fitting Random Forest Classifier (50 Trees)...")
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_resampled, y_train_resampled)
    joblib.dump(rf_model, os.path.join(models_dir, 'rf_model.joblib'))
    print("   [+] Saved rf_model.joblib")
    
    # 7. Fit and Persist XGBoost
    print("-> Fitting Fast XGBoost (Hist-tree method)...")
    xgb_model = XGBClassifier(tree_method='hist', random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train_resampled, y_train_resampled)
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgb_model.joblib'))
    print("   [+] Saved xgb_model.joblib")
    
    print("\nSUCCESS: All models and transformers have been saved to 'd:/final_aics/models/'")

if __name__ == "__main__":
    build_and_save_models()
