# src/validate_hallucinations.py
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath('src'))
from models import FedOmniMedFusion
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score

def print_hospital_report(hosp, df, model, device):
    print(f"\n" + "="*80)
    print(f"🏥 VALIDATION FOR HOSPITAL {hosp} (N={len(df)} Patients)")
    print("="*80)
    
    if hosp == 'A':
        masked_fields = "Vision (X-Ray), Blood Labs (CBC), RT-PCR"
    else:
        masked_fields = "Blood Labs (CBC), RT-PCR (Vision unavailable in this silo)"
        
    print(f"🟢 INPUT DATA (Given to AI)  : Age, Sex, Offset, Oxygen Need, ICU Status")
    print(f"🔴 MASKED DATA (Hidden)      : {masked_fields}")
    print("-" * 80)

    # Standardize vitals
    AGE_MEAN, AGE_STD = 52.0, 17.0
    OFFSET_MEAN, OFFSET_STD = 8.0, 8.0

    y_true_pcr, y_pred_pcr = [], []
    y_true_leuko, y_pred_leuko = [], []
    y_true_neutro, y_pred_neutro = [], []
    y_true_lympho, y_pred_lympho = [], []
    vision_correct, vision_total = 0, 0
    
    case_studies = []

    # Sort dataset to guarantee positive COVID cases appear in the 10 demo samples
    if 'RT_PCR_positive' in df.columns:
        df = df.sort_values(by='RT_PCR_positive', ascending=False).reset_index(drop=True)

    for idx, row in df.iterrows():
        # Inputs
        age, sex, offset = row['age'], row['sex'], row['offset']
        
        s_age = (age - AGE_MEAN) / AGE_STD
        s_offset = (offset - OFFSET_MEAN) / OFFSET_STD
        
        # Mask everything except vitals
        x_vitals = torch.tensor([[s_age, sex, s_offset, 
                                  row['intubated'], row['intubation_present'], row['needed_supplemental_O2'],
                                  row['went_icu'], row['in_icu'], 
                                  0.0, 0.0, 0.0]], dtype=torch.float32, device=device)

        with torch.no_grad():
            estimated = model.hallucinate(x_vitals=x_vitals, device=device)

        patient_case = {"input": f"Age:{age:.0f}, Sex:{int(sex)}, Offset:{offset:.0f}", "real": {}, "pred": {}}

        # Process Blood
        real_leuko, real_neutro, real_lympho = row.get('leukocyte_count'), row.get('neutrophil_count'), row.get('lymphocyte_count')
        if pd.notnull(real_leuko) and pd.notnull(real_neutro) and pd.notnull(real_lympho) and 'blood' in estimated:
            est_b = estimated['blood']
            
            p_leuko = est_b['leukocyte_scaled'] * 15.0
            p_neutro = est_b['neutrophil_scaled'] * 10.5
            p_lympho = est_b['lymphocyte_scaled'] * 5.0
            
            y_true_leuko.append(real_leuko); y_pred_leuko.append(float(p_leuko))
            y_true_neutro.append(real_neutro); y_pred_neutro.append(float(p_neutro))
            y_true_lympho.append(real_lympho); y_pred_lympho.append(float(p_lympho))
            
            patient_case["real"]["Blood"] = f"Leuko:{real_leuko:.1f}, Neutro:{real_neutro:.1f}"
            patient_case["pred"]["Blood"] = f"Leuko:{p_leuko:.1f}, Neutro:{p_neutro:.1f}"

        # Process PCR
        real_pcr = row.get('RT_PCR_positive')
        if pd.notnull(real_pcr):
            p_pcr = estimated['pcr']['positive_prob']
                
            y_true_pcr.append(real_pcr)
            y_pred_pcr.append(float(p_pcr))
            patient_case["real"]["PCR"] = "Positive" if real_pcr == 1 else "Negative"
            patient_case["pred"]["PCR"] = "Positive" if p_pcr > 0.5 else "Negative"

        # Process Vision
        real_finding = row.get('finding')
        if hosp == 'A' and pd.notnull(real_finding) and 'vision' in estimated:
            real_val = int(real_finding)
            
            top_finding = estimated['vision']['top_finding']
            is_normal_guess = (top_finding == "Normal / No Finding")
            
            real_val = int(real_finding)
            patient_case["real"]["Vision"] = "Clear" if real_val == 0 else "COVID"
            patient_case["pred"]["Vision"] = "Clear" if is_normal_guess else "Pathology"
            
            if (real_val == 0 and is_normal_guess) or (real_val == 1 and not is_normal_guess):
                vision_correct += 1
            vision_total += 1

        if len(case_studies) < 10:
            case_studies.append(patient_case)

    # Print Sample Cases
    print("\n[+] DETAILED CASE STUDIES (First 10 Patients):")
    for i, case in enumerate(case_studies):
        print(f"  Patient {i+1} Input  : {case['input']}")
        
        # Format Real
        r_str = " | ".join([f"{k} -> {v}" for k, v in case['real'].items()])
        # Format Pred
        p_str = " | ".join([f"{k} -> {v}" for k, v in case['pred'].items()])
        
        print(f"     Masked True: {r_str}")
        print(f"     AI Guessed : {p_str}\n")

    # Print Overall Metrics
    print(f"📊 --- HOSPITAL {hosp} OVERALL GENERATION PERFORMANCE ---")
    if vision_total > 0:
        print(f"  • Vision Inference Accuracy : {vision_correct/vision_total*100:.2f}% ({vision_correct}/{vision_total})")
    else:
        print(f"  • Vision Inference Accuracy : N/A (No Vision Data inside Hospital {hosp})")

    if len(y_true_pcr) > 0:
        acc = accuracy_score(y_true_pcr, [1.0 if p > 0.5 else 0.0 for p in y_pred_pcr])
        print(f"  • PCR Hallucination Accuracy: {acc*100:.2f}%")
        
    if len(y_true_leuko) > 0:
        mae_l = mean_absolute_error(y_true_leuko, y_pred_leuko)
        mae_n = mean_absolute_error(y_true_neutro, y_pred_neutro)
        print(f"  • Leukocyte MAE (Error)     : ±{mae_l:.2f} ×10⁹/L")
        print(f"  • Neutrophil MAE (Error)    : ±{mae_n:.2f} ×10⁹/L")

def main():
    model_path = 'saved_models/global_model_final.pth'
    if not os.path.exists(model_path):
        print(f"❌ Error: Model {model_path} not found.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FedOmniMedFusion().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    df_a = pd.read_csv('simulated_silos/Hospital_A/clinical_data.csv').sample(n=50, random_state=42).reset_index(drop=True)
    df_b = pd.read_csv('simulated_silos/Hospital_B/clinical_data.csv').sample(n=50, random_state=42).reset_index(drop=True)

    print_hospital_report('A', df_a, model, device)
    print_hospital_report('B', df_b, model, device)

if __name__ == "__main__":
    main()
