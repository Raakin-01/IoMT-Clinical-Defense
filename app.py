import os
import time
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Configure the Streamlit SOC Theme via Page Config natively
st.set_page_config(
    page_title="CRITICAL IoMT DDoS MONITOR",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark Mode SOC UI Styling Enhancements
st.markdown("""
<style>
    .critical-banner {
        background-color: #ff4d4d;
        color: white;
        padding: 20px;
        font-weight: 800;
        text-align: center;
        font-size: 24px;
        border-radius: 10px;
        margin-bottom: 20px;
        animation: blink 1.5s infinite;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Cache the ML Model Loading so users don't face constant re-loads
@st.cache_resource
def load_ml_assets():
    models_dir = 'd:/final_aics/models/'
    assets = {
        'scaler': None,
        'rf': None,
        'xgb': None,
        'features': []
    }
    
    if not os.path.exists(models_dir):
        return None
        
    try:
        assets['scaler'] = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
        assets['rf'] = joblib.load(os.path.join(models_dir, 'rf_model.joblib'))
        assets['xgb'] = joblib.load(os.path.join(models_dir, 'xgb_model.joblib'))
        assets['features'] = joblib.load(os.path.join(models_dir, 'feature_names.joblib'))
        return assets
    except Exception as e:
        return None

# Stylize the DataFrame using Pandas Styler
def color_suricata_flow(val):
    if val == "DDoS (Malicious)":
        return 'background-color: #8b0000; color: white' # Dark Red
    elif val == "Normal (Benign)":
        return 'background-color: #006400; color: white' # Dark Green
    return ''

def run():
    st.sidebar.title("🩺 IoMT Healthcare Network")
    menu = st.sidebar.radio("Navigation", ["Dashboard", "Live Prediction", "Model Metrics"])
    
    assets = load_ml_assets()

    if menu == "Dashboard":
        st.title("🛡️ Predictive DDoS Dashboard")
        st.markdown("Welcome to the **Healthcare Anomaly Intelligence platform**. Our Internet of Medical Things (IoMT) architecture runs continuously under the protection of machine-learned predictive modules evaluating flow-traffic vectors in real-time.")
        
        st.subheader("System Status")
        if assets:
            st.success("✅ ML Brain Active & Standardized")
            st.info("Operating Algorithmic Payload: `Fast Hist XGBoost` / `RandomForest 50-estimator Space`")
        else:
            st.error("❌ CRITICAL: No Pipeline Output Found. Please execute `python bridge_and_app.py` first to bind ML payload artifacts.")

    elif menu == "Live Prediction":
        st.title("📡 Live Prediction Terminal")
        st.write("Upload a captured PCAP CSV flow extraction to instantly scan the sequence against the IoMT ML baseline.")
        
        if not assets:
            st.error("Models not loaded. Refer to the Dashboard.")
            st.stop()
            
        model_choice = st.selectbox("Select Inference Engine:", ["XGBoost Classifier", "Random Forest Classifier"])
        active_model = assets['xgb'] if "XGBoost" in model_choice else assets['rf']
        
        uploaded_file = st.file_uploader("Upload Network Flow (CSV files)", type=['csv'])
        
        if uploaded_file is not None:
            with st.spinner("Decoding packets and extracting vector shapes..."):
                raw_df = pd.read_csv(uploaded_file)
                st.write(f"Inbound Flow Headers: `{len(raw_df)} Rows Segmented`")
                
                # Execute Preprocessing specifically matched to Train Schema
                df_test = raw_df.copy()
                
                # Ensure strictly required features are mapped
                for col in assets['features']:
                    if col not in df_test.columns:
                        # Create missing column filled with medians or 0 for inference
                        df_test[col] = 0 
                
                # Strip features not seen in training
                X_infer = df_test[assets['features']].copy()
                
                # Fill missing NaN
                for col in X_infer.columns:
                    if X_infer[col].dtypes in [float, int]:
                        X_infer[col] = X_infer[col].fillna(0)
                        
                start_scale_t = time.time()
                X_scaled = assets['scaler'].transform(X_infer)
                scale_time = time.time() - start_scale_t
                
                # Calculate actual real-time AI Prediction Latency
                start_t = time.time()
                predictions = active_model.predict(X_scaled)
                pred_latency = time.time() - start_t
                
                # Append labels visually
                result_df = raw_df.copy()
                result_df['Threat_Score'] = predictions
                result_df['Classification'] = result_df['Threat_Score'].map({0: "Normal (Benign)", 1: "DDoS (Malicious)"})
                
                # DDoS Threshold Alert Evaluation -> "If DDoS flows exceed 10%"
                ddos_ratio = np.sum(predictions) / len(predictions)
                
                # Layout metric stats
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Flows Processed", f"{len(raw_df):,}")
                col2.metric("Attack Probability %", f"{ddos_ratio * 100:.2f}%")
                col3.metric("Actual Mitigation Latency", f"{(pred_latency * 1000):.2f} ms")
                
                if ddos_ratio > 0.10:
                    st.markdown(f'<div class="critical-banner">🚨 CRITICAL: POTENTIAL DDOS ATTACK DETECTED IN HEALTHCARE NETWORK 🚨</div>', unsafe_allow_html=True)
                    st.audio("data:audio/wav;base64,UklGRnwGAABXQVZFRk1... (Silenced Ping Alarm placeholder)", format="audio/wav")   

                st.subheader("Flow Level Flagging Analysis")
                # Apply CSS highlighting mapping rule exclusively to label output using Styler
                st.dataframe(result_df.style.applymap(color_suricata_flow, subset=['Classification']))

    elif menu == "Model Metrics":
        st.title("📊 Architecture & Model Metrics")
        
        st.write("Visual artifacts generated historically during the latest `ddos_pipeline.py` pipeline optimization layer.")
        
        output_dir = 'd:/final_aics/'
        
        rf_cm = os.path.join(output_dir, 'random_forest_confusion_matrix.png')
        xgb_cm = os.path.join(output_dir, 'xgboost_confusion_matrix.png')
        
        rf_fi = os.path.join(output_dir, 'random_forest_feature_importance.png')
        xgb_fi = os.path.join(output_dir, 'xgboost_feature_importance.png')
        
        st.subheader("Random Forest Baseline")
        if os.path.exists(rf_cm) and os.path.exists(rf_fi):
            col1, col2 = st.columns(2)
            with col1:
                st.image(rf_cm, caption="Random Forest Test Confusion Matrix", use_container_width=True)
            with col2:
                st.image(rf_fi, caption="Top Ranked Red-Flag Features", use_container_width=True)
        else:
            st.warning("Random Forest artifacts missing.")
            
        st.write("---")
        
        st.subheader("XGBoost Hist Algorithm")
        if os.path.exists(xgb_cm) and os.path.exists(xgb_fi):
            col1, col2 = st.columns(2)
            with col1:
                st.image(xgb_cm, caption="XGBoost Test Confusion Matrix", use_container_width=True)
            with col2:
                st.image(xgb_fi, caption="Top Ranked Red-Flag Features", use_container_width=True)
        else:
            st.warning("XGBoost artifacts missing.")

if __name__ == "__main__":
    run()
