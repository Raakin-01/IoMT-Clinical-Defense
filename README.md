IoMT-DDoS-Predictor 🏥🛡️
Predictive Modeling for DDoS Mitigation in Critical Healthcare Networks

📌 Project Overview
This project implements a high-performance machine learning pipeline designed to protect Internet of Medical Things (IoMT) devices (e.g., heart monitors, MRI scanners) from Distributed Denial of Service (DDoS) attacks. Using the CIC-IoMT-2024 dataset, the system classifies network traffic in real-time and provides a professional SOC (Security Operations Center) dashboard for network administrators.

🚀 Key Features
Predictive Modeling: Uses optimized XGBoost and Random Forest classifiers to identify malicious traffic patterns.

Healthcare-First Logic: Optimized for a Low False Positive Rate (FPR) to ensure critical medical data is never accidentally blocked.

Real-Time Dashboard: A lightweight Flask-based SOC interface with dark-mode aesthetics and strobe-red attack alerts.

Performance Metrics: Real-time calculation of Mitigation Latency and Attack Probability.

🛠️ Tech Stack
Language: Python 3.x

ML Libraries: Scikit-Learn, XGBoost, Pandas, Joblib

Backend: Flask (Lightweight Localhost Server)

Frontend: HTML5/CSS3 (SOC Dark Theme)

Dataset: CIC-IoMT-2024

📂 Project Structure
Plaintext
├── ddos_pipeline.py    # Main ML pipeline (Training & Evaluation)
├── cool_server.py      # Flask SOC Dashboard & Real-time Inference
├── .gitignore          # Rules to exclude heavy CSV/ZIP data
├── models/             # Saved .joblib model files
└── README.md           # Project documentation
⚙️ Installation & Usage
Clone the repository:

Bash
git clone https://github.com/YOUR_USERNAME/IoMT-DDoS-Predictor.git
cd IoMT-DDoS-Predictor
Install Dependencies:

Bash
pip install flask pandas scikit-learn xgboost joblib
Train the Model:
Ensure your dataset is in the d:/final_aics/train/ path.

Bash
python ddos_pipeline.py
Launch the Dashboard:

Bash
python cool_server.py
Access the UI at: http://127.0.0.1:5000

Metric,Value
Model Type,Hist-Gradient Boosted Trees (XGBoost)
Target FPR,< 0.05%
Detection Latency,< 10ms per flow
Critical Alert Threshold,10% Malicious Traffic
