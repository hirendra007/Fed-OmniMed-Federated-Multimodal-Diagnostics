import pandas as pd
import numpy as np
import os
import shutil

# ─── Configuration ───────────────────────────────────────────────────────────
OUTPUT_DIR = "simulated_silos"

# The unified schema expected by the PyTorch Encoders
CLINICAL_COLS = [
    'age', 'sex', 'offset', 
    'intubated', 'intubation_present', 'needed_supplemental_O2', 
    'went_icu', 'in_icu', 'RT_PCR_positive',
    'leukocyte_count', 'neutrophil_count', 'lymphocyte_count'
]
TARGET_COLS = ['patientid', 'filename', 'finding'] + CLINICAL_COLS

def init_silos():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

# ─── Hospital A: Academic Center (Cohen Dataset) ─────────────────────────────
def process_hospital_a():
    print("Processing Hospital A (Cohen X-Ray Dataset)...")
    df = pd.read_csv("data/metadata.csv")
    
    # Map basic demographics
    df['sex'] = df['sex'].map({'M': 0.0, 'F': 1.0}).fillna(0.5)
    
    # Binary Y/N mapping
    binary_map = {'Y': 1.0, 'N': 0.0}
    for col in ['intubated', 'intubation_present', 'needed_supplemental_O2', 'went_icu', 'in_icu']:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0.0)
        else:
            df[col] = 0.0
            
    # Map PCR and Finding
    df['RT_PCR_positive'] = df['RT_PCR_positive'].map({'Y': 1.0, 'N': 0.0}).fillna(0.0)
    df['finding'] = df['finding'].apply(lambda x: 1.0 if 'COVID-19' in str(x) else 0.0)
    
    # ── Impute Missing Vitals ────────────────────────────────────────────────────────
    # The Cohen dataset (Hospital A) was scraped from medical journals, so not every 
    # paper reported the patient's Age or onset Offset. We simulate these missing values
    # by generating them from normal distributions matching the global average.
    np.random.seed(42)
    missing_age = df['age'].isnull()
    df.loc[missing_age, 'age'] = np.random.normal(52, 15, size=missing_age.sum()).clip(min=18, max=90)
    
    missing_offset = df['offset'].isnull()
    df.loc[missing_offset, 'offset'] = np.random.normal(7, 4, size=missing_offset.sum()).clip(min=0, max=21)

    # ── Generative Synthetic Blood Insertion ─────────────────────────────────────────
    # Hospital A lacks CBC blood tests natively. To create robust data for validation,
    # we inject synthetic normal distributions directly modeled off Hospital B metrics.
    # Leukocyte: μ=7.5, σ=2.5 | Neutrophil: μ=4.5, σ=2.0 | Lymphocyte: μ=2.0, σ=1.0
    df['leukocyte_count'] = np.random.normal(7.5, 2.5, size=len(df)).clip(min=0.1)
    df['neutrophil_count'] = np.random.normal(4.5, 2.0, size=len(df)).clip(min=0.1)
    df['lymphocyte_count'] = np.random.normal(2.0, 1.0, size=len(df)).clip(min=0.1)

    # Keep only relevant columns and fill any other missing labs with NaN
    for col in TARGET_COLS:
        if col not in df.columns:
            df[col] = np.nan
            
    df_out = df[TARGET_COLS].copy()
    save_silo(df_out, "Hospital_A", "Academic Medical Center")


# ─── Hospital B: Urban Clinic (Einstein Dataset) ─────────────────────────────
def process_hospital_b():
    print("Processing Hospital B (Einstein Blood Labs Dataset)...")
    df = pd.read_excel("data/dataset.xlsx")
    
    # Filter: Keep only patients who actually had blood work done
    df = df.dropna(subset=['Leukocytes', 'Lymphocytes', 'Neutrophils'], how='all')
    
    # Schema Alignment
    df['patientid'] = df['Patient ID']
    df['filename'] = None  # No images
    df['finding'] = df['SARS-Cov-2 exam result'].map({'positive': 1.0, 'negative': 0.0})
    df['RT_PCR_positive'] = df['finding'] # All findings here are derived from PCR
    
    # Age quantile to approximate age (Quantile * 5 years roughly aligns the distribution)
    df['age'] = df['Patient age quantile'] * 5.0
    df['sex'] = 0.5 # Einstein dataset omits sex for privacy, use 0.5 (unknown)
    df['offset'] = 7.0 # Median imputation for onset days
    
    # Triage mappings
    df['went_icu'] = df['Patient addmited to intensive care unit (1=yes, 0=no)']
    df['in_icu'] = df['went_icu']
    df['needed_supplemental_O2'] = df['Patient addmited to semi-intensive unit (1=yes, 0=no)']
    df['intubated'] = 0.0
    df['intubation_present'] = 0.0
    
    # Un-Standardize blood labs: Einstein data is Z-scored. We reverse this to raw values 
    # so Hospital B's distribution matches the global norm.
    # Formula: raw = (z_score * standard_deviation) + mean
    df['leukocyte_count'] = (df['Leukocytes'] * 2.5) + 7.5
    df['neutrophil_count'] = (df['Neutrophils'] * 2.0) + 4.5
    df['lymphocyte_count'] = (df['Lymphocytes'] * 1.0) + 2.0
    
    # Clip to prevent negative blood counts from arbitrary z-scores
    df['leukocyte_count'] = df['leukocyte_count'].clip(lower=0.1)
    df['neutrophil_count'] = df['neutrophil_count'].clip(lower=0.1)
    df['lymphocyte_count'] = df['lymphocyte_count'].clip(lower=0.1)
    
    df_out = df[TARGET_COLS].copy()
    save_silo(df_out, "Hospital_B", "Urban Clinic (Blood Labs & PCR)")


# ─── Hospital C: Community Clinic (Mexican Gov Dataset) ──────────────────────
def process_hospital_c():
    print("Processing Hospital C (Mexican Gov Triage Dataset)...")
    # Read only needed columns to save memory
    cols_to_use = ['id', 'covid_res', 'age', 'sex', 'entry_date', 'date_symptoms', 'intubed', 'pneumonia', 'icu']
    df = pd.read_csv("data/covid.csv", usecols=cols_to_use)
    
    # The Mexican dataset is massive (1M+ rows). We must sample it to prevent it from 
    # destroying the federated learning balance. Let's take ~2000 balanced rows.
    df_covid = df[df['covid_res'] == 1].sample(n=1000, random_state=42)
    df_normal = df[df['covid_res'] == 2].sample(n=1000, random_state=42)
    df = pd.concat([df_covid, df_normal]).sample(frac=1.0).reset_index(drop=True)
    
    # Schema Alignment
    df['patientid'] = df['id']
    df['filename'] = None
    df['finding'] = df['covid_res'].map({1: 1.0, 2: 0.0, 3: 0.0}) # 1 is positive, 2/3 are negative/pending
    
    df['age'] = df['age']
    # Mexican dataset: 1 = Female, 2 = Male -> Model: 1.0 = Female, 0.0 = Male
    df['sex'] = df['sex'].map({1: 1.0, 2: 0.0}) 
    
    # Calculate days since onset
    entry = pd.to_datetime(df['entry_date'], errors='coerce', format='%d-%m-%Y')
    onset = pd.to_datetime(df['date_symptoms'], errors='coerce', format='%d-%m-%Y')
    df['offset'] = (entry - onset).dt.days.clip(lower=0.0).fillna(3.0)
    
    # Mexican dataset specific values: 1=Yes, 2=No, 97/98/99=Missing/Not Applicable
    def map_mexican_binary(val):
        return 1.0 if val == 1 else 0.0
        
    df['intubated'] = df['intubed'].apply(map_mexican_binary)
    df['intubation_present'] = df['intubated']
    df['needed_supplemental_O2'] = df['pneumonia'].apply(map_mexican_binary)
    df['went_icu'] = df['icu'].apply(map_mexican_binary)
    df['in_icu'] = df['went_icu']
    
    # Missing Modalities for Hospital C
    df['RT_PCR_positive'] = np.nan
    df['leukocyte_count'] = np.nan
    df['neutrophil_count'] = np.nan
    df['lymphocyte_count'] = np.nan
    
    df_out = df[TARGET_COLS].copy()
    save_silo(df_out, "Hospital_C", "Community Clinic (Vitals Only)")


def save_silo(df, name, description):
    path = os.path.join(OUTPUT_DIR, name)
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, "clinical_data.csv"), index=False)
    
    print(f"✅ Created {name}: {description}")
    print(f"   Records: {len(df)}")
    print(f"   COVID+: {int(df['finding'].sum())}")
    print(f"   Missing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}\n")

if __name__ == "__main__":
    init_silos()
    process_hospital_a()
    process_hospital_b()
    process_hospital_c()