import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_and_sample_data(data_dir, samples_per_file=2000, random_state=42):
    """Reads all CSV files from a directory, assigns labels, and samples rows."""
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}. Please check your dataset path.")
    
    df_list = []
    
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Locating CSVs in {data_dir}...")
    for file in csv_files:
        filename = os.path.basename(file).lower()
        # Label 0 for Benign, 1 for DDoS/Malicious
        label = 0 if 'benign' in filename else 1
        
        try:
            # Read all to memory briefly then sample
            df = pd.read_csv(file)
            
            # Smart sampling
            if len(df) > samples_per_file:
                df = df.sample(n=samples_per_file, random_state=random_state)
            
            df['Label'] = label
            df_list.append(df)
            
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Loaded {len(df_list)} files.")
    
    # Combine datasets
    full_df = pd.concat(df_list, ignore_index=True)
    return full_df

def preprocess_data(df):
    """Handles missing values, drops zero variance columns, and prepares features."""
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Starting preprocessing on {len(df)} rows...")
    
    X = df.drop(columns=['Label'])
    y = df['Label'].copy()

    # Drop columns with > 50% missing values
    missing_pct = X.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.5].index
    if len(cols_to_drop) > 0:
        print(f"Dropping high-missing features: {list(cols_to_drop)}")
        X = X.drop(columns=cols_to_drop)
    
    # Identify numeric and non-numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns

    # Fill remaining missing values with median
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())
        
    # Drop non-numeric features for lightweight ML
    if len(non_numeric_cols) > 0:
        print(f"Dropping non-numeric features: {list(non_numeric_cols)}")
        X = X.drop(columns=non_numeric_cols)
        
    # Drop zero variance columns
    variance_selector = VarianceThreshold(threshold=0.0)
    variance_selector.fit(X)
    constant_columns = [c for c in X.columns if c not in X.columns[variance_selector.get_support()]]
    if len(constant_columns) > 0:
        print(f"Dropping zero-variance features: {constant_columns}")
        X = X.drop(columns=constant_columns)
        
    return X, y

def evaluate_and_plot(model, model_name, X_test, y_test, output_dir="."):
    """Evaluates model, calculates metrics (including FPR), and plots Confusion Matrix."""
    y_pred = model.predict(X_test)
    
    print(f"\n{'='*30}\n{model_name} Evaluation\n{'='*30}")
    
    cm = confusion_matrix(y_test, y_pred)
    # Binary classification expected. If multiple classes exist, this ravel() will fail. 
    # But we explicitly forced binary labels (0/1).
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = f1_score(y_test, y_pred)
    
    print(f"F1-Score: {f1:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f} (Lower is better!)")
    print(f"False Positives: {fp} (Legitimate connections blocked)")
    print(f"False Negatives: {fn} (DDoS attacks missed)\n")
    print(classification_report(y_test, y_pred, target_names=["Benign", "DDoS"]))
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Benign", "DDoS"], 
                yticklabels=["Benign", "DDoS"])
    plt.title(f"{model_name} - Confusion Matrix\nFPR: {fpr:.4f} | F1: {f1:.4f}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png"))
    plt.show(block=False)
    print(f"Saved {model_name} Confusion Matrix plot.")

def plot_feature_importance(model, feature_names, model_name, output_dir="."):
    """Plots the top 15 most important features for interpretability."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print(f"{model_name} does not have feature_importances_. Skipping plot.")
        return
        
    indices = np.argsort(importances)[::-1][:15] # Top 15
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_importances, y=top_features, palette='viridis')
    plt.title(f"{model_name} - Top 15 Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_feature_importance.png"))
    plt.show(block=False)
    print(f"Saved {model_name} Feature Importance plot.")

def main():
    data_dir = 'd:/final_aics/train/'
    output_dir = 'd:/final_aics/'
    
    print("--- 1. Data Collection & Smart Sampling ---")
    df = load_and_sample_data(data_dir, samples_per_file=2000)
    
    print("--- 2. Efficient Preprocessing ---")
    X, y = preprocess_data(df)
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    print("--- 3. Training and Optimizing The Models ---")
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Training Random Forest...")
    rf_model.fit(X_train_resampled, y_train_resampled)
    
    evaluate_and_plot(rf_model, "Random Forest", X_test_scaled, y_test, output_dir)
    plot_feature_importance(rf_model, feature_names, "Random Forest", output_dir)
    
    xgb_model = XGBClassifier(tree_method='hist', random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
    print(f"\n[{pd.Timestamp.now().strftime('%H:%M:%S')}] Training XGBoost...")
    xgb_model.fit(X_train_resampled, y_train_resampled)
    
    evaluate_and_plot(xgb_model, "XGBoost", X_test_scaled, y_test, output_dir)
    plot_feature_importance(xgb_model, feature_names, "XGBoost", output_dir)
    print("\nPipeline execution complete! Displaying all generated PNG artifacts on screen...")
    plt.show()

if __name__ == "__main__":
    main()
