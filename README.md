# Fed-OmniMed-Federated-Multimodal-Diagnostics

Fed-OmniMed is a robust, privacy-preserving federated learning system designed for multimodal medical diagnostics (e.g., COVID-19 detection). It is built using **PyTorch** and **Flower (flwr)**, enabling collaborative model training across disparate healthcare facilities without centralizing sensitive patient data.

## 🌟 Key Features

*   **Federated Learning**: Uses the FedProx strategy to train a global model collaboratively across multiple "silos" (hospitals) while keeping raw data local.
*   **Multimodal Architecture**: Processes diverse data types:
    *   Vision (Chest X-Rays) via ResNet-18
    *   Vitals (Age, Sex, SpO2, Temperature, etc.)
    *   Blood Labs (CBC: Leukocytes, Neutrophils, Lymphocytes)
    *   RT-PCR Results
*   **Modality-Aware Gating (MAG)**: A dynamic gating mechanism that weights modalities based on their availability, allowing the model to act as a generalist.
*   **Feature Imputation Network (FIN)**: Learns fallback representations for missing modalities. This ensures robust performance even in low-tech environments (e.g., community clinics lacking X-ray capabilities).
*   **GPU Acceleration**: Fully supports NVIDIA CUDA for accelerated local training, global aggregation, and real-time inference.
*   **Interactive Diagnostic UI**: A Streamlit dashboard for real-time inference using the global federated model, featuring a "Hallucination Inspector" to visualize FIN confidence and modality attribution.

---

## 🏛️ Simulated Healthcare Tiers

The project simulates three distinct healthcare environments to test the robustness of the multimodal architecture:

1.  **Hospital A (Academic Medical Center)**: High-resource. Access to all modalities (X-Rays, Vitals, Full Labs, PCR).
2.  **Hospital B (Urban Clinic)**: Medium-resource. Access to Vitals, Blood Labs, and PCR, but *no Imaging (X-Ray)*.
3.  **Hospital C (Community Clinic)**: Low-resource. Access only to Core Vitals. *No Imaging, Labs, or PCR*.

---

## ⚙️ Setup Instructions

### Prerequisites
*   Python 3.9+
*   *(Optional but recommended)* NVIDIA GPU with CUDA Toolkit installed.

### 1. Environment Setup

Clone the repository and create a virtual environment:

```powershell
git clone https://github.com/hirendra007/Fed-OmniMed-Federated-Multimodal-Diagnostics.git
cd Fed-OmniMed-Federated-Multimodal-Diagnostics
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

> **GPU Support**: If you have an NVIDIA GPU, ensure you install the CUDA-enabled version of PyTorch. Refer to `GPU_SETUP.md` for detailed instructions.

### 3. Prepare the Data

Generate the simulated hospital silos from the raw dataset:

```powershell
python src/data_loader.py
```
*This will create a `simulated_silos/` directory containing distributed data for Hospitals A, B, and C.*

---

## 🚀 Running the Federated Network

To run a full federated training simulation, you will need to open multiple terminal windows (one for the server and one for each client). Ensure your virtual environment (`.\venv\Scripts\activate`) is active in **every** terminal.

### Step 1: Start the Server
In Terminal 1, start the Flower server. It will wait for clients to connect.
```powershell
python src/server.py
```

### Step 2: Start the Clients
Open three new terminals (Terminal 2, 3, and 4) and start the clients representing each hospital tier:

*Terminal 2 (Hospital A):*
```powershell
python src/client.py --id A
```

*Terminal 3 (Hospital B):*
```powershell
python src/client.py --id B
```

*Terminal 4 (Hospital C):*
```powershell
python src/client.py --id C
```

The server will coordinate training across the clients for the configured number of rounds. Once complete, the final global model weights will be saved to `saved_models/global_model_final.pth`.

---

## 📊 Evaluation & Metrics

After federated training completes, you can generate a comprehensive metrics report across all hospital tiers:

```powershell
python src/generate_full_metrics.py
```
This script evaluates the global model against the local test sets of each hospital, producing Accuracy, Precision, Recall, Specificity, F1-Score, and Confusion Matrices. Results are output to the console and saved in `final_metrics.json`.

---

## 🩺 Interactive Dashboard

Launch the Streamlit web interface to run live inferences using the trained global model:

```powershell
streamlit run web_ui/app.py
```

**Dashboard Features:**
*   Simulate patient input across different hospital tiers.
*   Adjust diagnostic sensitivity thresholds.
*   View clinical verdicts (Risk Probability).
*   Inspect the Feature Imputation Network (FIN) to see how missing modalities are handled via the Hallucination Inspector.

---

## 📁 Project Structure

```
Fed-OmniMed_Project/
│
├── data/                       # Raw datasets (metadata.csv, covid.csv, etc.)
├── simulated_silos/            # Generated silo data (created by data_loader.py)
├── saved_models/               # Directory for saved global model weights
│
├── src/                        # Source Code
│   ├── client.py               # FL Client & PyTorch Training Loop
│   ├── data_loader.py          # Data distribution & silo generation
│   ├── export_diagnostics.py   # Diagnostic utilities
│   ├── generate_full_metrics.py# Evaluation & Metrics reporting
│   ├── models.py               # PyTorch Architecture (MAG, FIN, ResNet, etc.)
│   └── server.py               # FL Server & Custom FedProx Strategy
│
├── web_ui/                     # Interactive Interface
│   └── app.py                  # Streamlit application
│
├── requirements.txt            # Python dependencies
├── final_metrics.json          # Exported metrics report
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```
