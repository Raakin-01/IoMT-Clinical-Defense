import os
import time
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string

# Import the original, untouched data preprocessor
from ddos_pipeline import preprocess_data

app = Flask(__name__)

# --- LOAD MODELS ---
# We assume bridge_and_app.py has already successfully cached these to disk
MODELS_DIR = 'd:/final_aics/models/'
try:
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgb_model.joblib'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.joblib'))
    MODEL_FPR_RATING = "0.0425" # Static from earlier evaluation
except Exception as e:
    print(f"ERROR loading ML components: {e}\\nPlease run bridge_and_app.py first.")
    # Provide fallbacks purely to prevent crashing without bridge execution
    scaler, xgb_model, feature_names = None, None, []
    MODEL_FPR_RATING = "ERR"

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CRITICAL IoMT DDoS MONITOR</title>
    <style>
        :root {
            --bg-color: #0f1115;
            --card-bg: #1e2129;
            --text-color: #ffffff;
            --accent-green: #00ff00;
            --accent-red: #ff0000;
            --glow-color: rgba(0, 255, 0, 0.4);
            --border-color: #333;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }

        header {
            width: 100%;
            background-color: var(--card-bg);
            border-bottom: 2px solid var(--border-color);
            padding: 20px 0;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
            margin-bottom: 40px;
        }

        h1 {
            margin: 0;
            font-size: 2.5rem;
            letter-spacing: 2px;
            color: var(--text-color);
        }
        
        .subtitle {
            margin-top: 5px;
            color: #888;
            font-size: 1rem;
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        .container {
            width: 80%;
            max-width: 1000px;
            display: flex;
            flex-direction: column;
            gap: 30px;
        }

        /* Drag and Drop Zone */
        .upload-zone {
            background-color: var(--card-bg);
            border: 2px dashed #555;
            border-radius: 8px;
            padding: 50px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .upload-zone:hover, .upload-zone.dragover {
            border-color: var(--accent-green);
            background-color: rgba(0, 255, 0, 0.05);
        }

        .upload-zone h3 {
            margin: 0 0 10px 0;
            color: #aaa;
        }

        .upload-zone p {
            color: #666;
            font-size: 0.9rem;
        }
        
        #fileInput {
            display: none;
        }

        /* Glowing Button */
        button {
            background-color: transparent;
            color: var(--accent-green);
            border: 2px solid var(--accent-green);
            border-radius: 5px;
            padding: 15px 30px;
            font-size: 1.2rem;
            text-transform: uppercase;
            font-weight: bold;
            letter-spacing: 1px;
            cursor: pointer;
            transition: all 0.3s ease;
            outline: none;
            box-shadow: 0 0 10px var(--glow-color);
            align-self: center;
            width: 50%;
            margin: 0 auto;
        }

        button:hover {
            background-color: var(--accent-green);
            color: var(--bg-color);
            box-shadow: 0 0 20px var(--glow-color), 0 0 40px var(--glow-color);
        }
        
        button:disabled {
            border-color: #555;
            color: #555;
            box-shadow: none;
            cursor: not-allowed;
            background-color: transparent;
        }

        /* Results Section */
        #resultsSection {
            display: none;
            flex-direction: column;
            gap: 20px;
            animation: fadeIn 0.5s;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Attack Banner */
        .critical-banner {
            display: none;
            background-color: var(--accent-red);
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 2rem;
            font-weight: 800;
            text-transform: uppercase;
            border-radius: 8px;
            animation: strobe 1s infinite;
            box-shadow: 0 0 20px rgba(255, 0, 0, 0.5);
        }

        @keyframes strobe {
            0%, 100% { background-color: var(--accent-red); color: white; transform: scale(1); }
            50% { background-color: #8b0000; color: #ccc; transform: scale(1.02); }
        }
        
        .clean-banner {
            display: none;
            background-color: var(--card-bg);
            border: 2px solid var(--accent-green);
            color: var(--accent-green);
            padding: 20px;
            text-align: center;
            font-size: 2rem;
            font-weight: 800;
            text-transform: uppercase;
            border-radius: 8px;
            box-shadow: 0 0 15px rgba(0, 255, 0, 0.2);
        }

        /* Metric Cards Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }

        .metric-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .metric-card.alert {
            border-color: rgba(255, 0, 0, 0.5);
            background-color: rgba(255, 0, 0, 0.05);
        }

        .metric-title {
            color: #888;
            font-size: 0.9rem;
            text-transform: uppercase;
            margin-bottom: 10px;
            letter-spacing: 1px;
        }

        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--text-color);
        }
        
        .metric-value.red { color: var(--accent-red); text-shadow: 0 0 10px rgba(255,0,0,0.5); }
        .metric-value.green { color: var(--accent-green); text-shadow: 0 0 10px rgba(0,255,0,0.5); }

        #loading {
            display: none;
            text-align: center;
            color: var(--accent-green);
            font-size: 1.2rem;
            margin-top: 10px;
            animation: pulse 1s infinite alternate;
        }
        
        @keyframes pulse {
            0% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>

    <header>
        <h1>IoMT DDoS Mitigation Dashboard</h1>
        <div class="subtitle">Security Operations Center - Command Node</div>
    </header>

    <div class="container">
        
        <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
            <h3 id="uploadText">Drag & Drop Network CSV Profile Here</h3>
            <p id="subUploadText">or click to browse local files</p>
            <input type="file" id="fileInput" accept=".csv">
        </div>

        <button id="analyzeBtn" onclick="analyzeTraffic()" disabled>Waiting for Payload...</button>
        <div id="loading">DECRYPTING AND ANALYZING TRAFFIC VECTORS...</div>

        <div id="resultsSection">
            <div id="attackBanner" class="critical-banner">CRITICAL: IOMT NETWORK UNDER ATTACK</div>
            <div id="cleanBanner" class="clean-banner">NETWORK STATUS: CLEAN</div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">Total Flows Processed</div>
                    <div class="metric-value" id="valTotalFlows">-</div>
                </div>
                <div class="metric-card" id="cardAttack">
                    <div class="metric-title">Attack Probability</div>
                    <div class="metric-value" id="valAttackProb">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Model FPR Rating</div>
                    <div class="metric-value" id="valFpr">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Mitigation Latency</div>
                    <div class="metric-value" id="valLatency">-</div>
                </div>
            </div>
        </div>
        
    </div>

    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const uploadText = document.getElementById('uploadText');
        const subUploadText = document.getElementById('subUploadText');
        
        let selectedFile = null;

        // Drag and Drop Handlers
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            
            if (e.dataTransfer.files.length > 0) {
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });

        function handleFileSelect(file) {
            if (file.name.endsWith('.csv')) {
                selectedFile = file;
                uploadText.textContent = `TARGET ACQUIRED: ${file.name}`;
                uploadText.style.color = 'var(--accent-green)';
                subUploadText.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB Payload Ready`;
                
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = "Analyze Traffic";
                
                // Reset UI
                document.getElementById('resultsSection').style.display = 'none';
            } else {
                uploadText.textContent = "INVALID PAYLOAD FORMAT";
                uploadText.style.color = 'var(--accent-red)';
                subUploadText.textContent = "Please upload a valid .csv flow capture.";
                selectedFile = null;
                analyzeBtn.disabled = true;
            }
        }

        async function analyzeTraffic() {
            if (!selectedFile) return;

            const formData = new FormData();
            formData.append('file', selectedFile);

            analyzeBtn.style.display = 'none';
            document.getElementById('loading').style.display = 'block';
            document.getElementById('resultsSection').style.display = 'none';
            
            // Reset Banners
            document.getElementById('attackBanner').style.display = 'none';
            document.getElementById('cleanBanner').style.display = 'none';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert("Analysis Failed: " + data.error);
                } else {
                    displayResults(data);
                }
            } catch (error) {
                console.error("Transmission Error:", error);
                alert("Network communication severed with SOC Backend.");
            } finally {
                analyzeBtn.style.display = 'block';
                analyzeBtn.textContent = "RE-ANALYZE NODE";
                document.getElementById('loading').style.display = 'none';
            }
        }

        function displayResults(data) {
            // Un-hide results
            document.getElementById('resultsSection').style.display = 'flex';
            
            // Populate metrics
            document.getElementById('valTotalFlows').textContent = data.total_flows.toLocaleString();
            document.getElementById('valFpr').textContent = data.fpr;
            document.getElementById('valLatency').textContent = data.latency_ms.toFixed(2) + " ms";
            
            const probElem = document.getElementById('valAttackProb');
            const cardAttack = document.getElementById('cardAttack');
            
            probElem.textContent = (data.attack_probability * 100).toFixed(2) + "%";
            
            // Evaluated Threshold logic
            if (data.attack_probability > 0.10) {
                document.getElementById('attackBanner').style.display = 'block';
                probElem.className = "metric-value red";
                cardAttack.className = "metric-card alert";
            } else {
                document.getElementById('cleanBanner').style.display = 'block';
                probElem.className = "metric-value green";
                cardAttack.className = "metric-card";
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(HTML_CONTENT)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    try:
        # Load raw payload stream
        raw_df = pd.read_csv(file)
        
        # Zero Modification Rule: Use the same preprocess_data logic imported untouched 
        # Note: preprocess_data internally expects a 'Label' column purely to separate it.
        # Live flow sets will NOT have a precise 'Label'. We will temporarily inject a dummy 
        # one so the original untouched function executes flawlessly without breaking.
        if 'Label' not in raw_df.columns:
            raw_df['Label'] = 0 
            
        X_test_clean, _ = preprocess_data(raw_df)
        
        # Explicit alignment with trained features
        for col in feature_names:
            if col not in X_test_clean.columns:
                X_test_clean[col] = 0
                
        X_infer = X_test_clean[feature_names].copy()
        
        # Fill missing NaN (if new completely null cols were spawned dynamically, e.g. zero variance drop)
        for col in X_infer.columns:
            if X_infer[col].dtypes in [float, int]:
                X_infer[col] = X_infer[col].fillna(0)

        # Pre-infer mapping bounds
        X_scaled = scaler.transform(X_infer)
        
        # Measure EXACT Inference Latency via perf_counter per the request
        start_time = time.perf_counter()
        predictions = xgb_model.predict(X_scaled)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        
        # Evaluate Probability Density
        total_flows = len(predictions)
        ddos_ratio = float(np.sum(predictions) / total_flows) if total_flows > 0 else 0.0

        return jsonify({
            'status': 'success',
            'total_flows': total_flows,
            'attack_probability': ddos_ratio,
            'latency_ms': latency_ms,
            'fpr': MODEL_FPR_RATING
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Execute Local Flask App cleanly. Port 5000 is default.
    app.run(debug=True, port=5000, host="0.0.0.0")
