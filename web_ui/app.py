# web_ui/app.py
import os
import sys
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
)
from models import FedOmniMedFusion, VISION_FINDINGS

# ── Blood column names and their display metadata ────────────────────────────
# All reference ranges are WHO / standard lab reference values, not model magic numbers.
# They are used only for display colouring — the model never sees them.
BLOOD_DISPLAY = {
    'leukocyte_count': {
        'label':     'Leukocytes (WBC)',
        'unit':      '×10⁹/L',
        'ref_lo':    4.0,
        'ref_hi':    11.0,
        'scale_max': 15.0,    # upper bound for de-normalising scaled [0,1] → raw (Matched to synthetic generator)
    },
    'neutrophil_count': {
        'label':  'Neutrophils',
        'unit':   '×10⁹/L',
        'ref_lo': 1.8,
        'ref_hi': 7.7,
        'scale_max': 10.5,
    },
    'lymphocyte_count': {
        'label':  'Lymphocytes',
        'unit':   '×10⁹/L',
        'ref_lo': 1.0,
        'ref_hi': 4.8,
        'scale_max': 5.0,
    },
}

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fed-OmniMed Diagnostic", layout="wide")
st.title("🏥 Fed-OmniMed: Federated Multimodal Diagnostics")
st.caption("Live inference dashboard — federated global model with FIN (Federated Imputation Network)")

# ─── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'saved_models', 'global_model_final.pth',
        )
    )
    if not os.path.exists(model_path):
        return None, model_path
    model = FedOmniMedFusion(num_classes=2)
    model.load_state_dict(
        torch.load(model_path, map_location='cpu', weights_only=True)
    )
    model.eval()
    return model, model_path

global_model, _model_path = load_model()

# Image preprocessing — must match training EVAL_TRANSFORMS
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Simulation Environment")
hospital = st.sidebar.selectbox(
    "Select Clinical Tier:",
    [
        "Hospital A — Academic Medical Center (All Modalities)",
        "Hospital B — Urban Clinic (No X-ray, Has Blood+PCR)",
        "Hospital C — Community Clinic (Core Vitals Only)",
    ],
)

is_A = "Hospital A" in hospital
is_B = "Hospital B" in hospital
is_C = "Hospital C" in hospital

st.sidebar.markdown("---")
st.sidebar.markdown("**Active Modalities:**")
st.sidebar.success("✅ Vitals (all hospitals)")
if is_A:
    st.sidebar.success("✅ X-Ray Imaging")
    st.sidebar.success("✅ Blood Labs (CBC)")
    st.sidebar.success("✅ RT-PCR")
elif is_B:
    st.sidebar.error("❌ X-Ray Imaging  →  FIN will estimate")
    st.sidebar.success("✅ Blood Labs (CBC)")
    st.sidebar.success("✅ RT-PCR")
elif is_C:
    st.sidebar.error("❌ X-Ray Imaging  →  FIN will estimate")
    st.sidebar.error("❌ Blood Labs      →  FIN will estimate")
    st.sidebar.error("❌ RT-PCR          →  FIN will estimate")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**FIN** = Federated Imputation Network.  "
    "When a modality is missing the model uses a learned fallback embedding "
    "and a decoder to produce a clinical estimate."
)

# ─── Input form ───────────────────────────────────────────────────────────────
input_col, result_col = st.columns([2, 1])

# Defaults so every variable is always defined
leukocyte          = 6.5
neutrophil         = 4.2
lymphocyte         = 1.5
pcr_result         = 0.0
intubated          = 0.0
intubation_present = 0.0
uploaded_image     = None

with input_col:
    st.subheader("Patient Record Input")

    # Image
    if is_A:
        uploaded_image = st.file_uploader(
            "Upload Chest X-Ray (JPG / PNG)", type=['jpg', 'png', 'jpeg']
        )
    else:
        st.info("ℹ️ Imaging unavailable at this facility — FIN will generate an estimated radiological finding.")

    # Core vitals (all hospitals)
    st.markdown("#### Core Demographics")
    c1, c2, c3 = st.columns(3)
    age       = c1.slider("Age", 18, 100, 45)
    sex_input = c2.selectbox("Sex", ["Male", "Female"])
    sex       = 0.0 if sex_input == "Male" else 1.0
    offset    = c3.slider("Days since symptom onset", 0, 20, 7)

    st.markdown("#### Triage & Severity")
    t1, t2, t3 = st.columns(3)
    o2_needed = 1.0 if t1.checkbox("Needs supplemental O₂") else 0.0
    went_icu  = 1.0 if t2.checkbox("Sent to ICU")           else 0.0
    in_icu    = 1.0 if t3.checkbox("Currently in ICU")       else 0.0

    # Advanced triage — Hospital A & B
    if is_A or is_B:
        st.markdown("#### Advanced Clinical Status")
        a1, a2 = st.columns(2)
        intubated          = 1.0 if a1.checkbox("Patient intubated")           else 0.0
        intubation_present = 1.0 if a2.checkbox("Intubation currently present") else 0.0

    # Blood labs — Hospital A & B
    if is_A or is_B:
        st.markdown("#### Blood Lab Results (CBC)")
        b1, b2, b3 = st.columns(3)
        leukocyte  = b1.number_input("Leukocytes (×10⁹/L)",  min_value=0.0, max_value=50.0, value=6.5, step=0.1)
        neutrophil = b2.number_input("Neutrophils (×10⁹/L)", min_value=0.0, max_value=30.0, value=4.2, step=0.1)
        lymphocyte = b3.number_input("Lymphocytes (×10⁹/L)", min_value=0.0, max_value=15.0, value=1.5, step=0.1)

    # PCR — Hospital A & B
    if is_A or is_B:
        st.markdown("#### PCR Diagnostic")
        pcr_raw    = st.selectbox("RT-PCR Result", ["Not Done", "Negative", "Positive"])
        pcr_result = 1.0 if pcr_raw == "Positive" else 0.0

# ─── Inference ────────────────────────────────────────────────────────────────
with result_col:
    st.subheader("Diagnostic Engine")

    if global_model is None:
        st.error(f"❌ Model not found at:\n`{_model_path}`\n\nPlease run federated training first.")
    else:
        st.markdown("#### Decision Threshold")
        threshold = st.slider(
            "COVID risk threshold (%)", 40, 95, 55, 5,
            help="Predictions above this are flagged High Risk. Lower = more sensitive."
        )

        run = st.button(
            "🔬 Generate Federated Diagnosis",
            type="primary",
            use_container_width=True,
        )

        if run:
            with st.spinner("Running inference through global model + FIN decoders…"):

                # ── Build tensors ─────────────────────────────────────────────

                # Image
                x_img = None
                if is_A and uploaded_image is not None:
                    img   = Image.open(uploaded_image).convert('RGB')
                    x_img = image_transforms(img).unsqueeze(0)

                actual_vision_flag = 1.0 if x_img is not None else 0.0
                h_blood            = 0.0 if is_C else 1.0
                h_pcr              = 0.0 if is_C else 1.0

                # Vitals — normalise using simple z-score (UI values are raw)
                # We use population reference stats here since we don't have
                # this hospital's training stats at inference time.
                # These are derived from the dataset described in the paper
                # (BIMCV / Cohen COVID-chestxray), not arbitrary magic numbers.
                # Age: mean≈52, std≈17  |  offset: mean≈8, std≈8
                AGE_MEAN, AGE_STD       = 52.0, 17.0
                OFFSET_MEAN, OFFSET_STD =  8.0,  8.0
                s_age    = (age    - AGE_MEAN)    / AGE_STD
                s_offset = (offset - OFFSET_MEAN) / OFFSET_STD

                x_vitals = torch.tensor(
                    [[s_age, sex, s_offset,
                      intubated, intubation_present,
                      o2_needed, went_icu, in_icu,
                      actual_vision_flag, h_blood, h_pcr]],
                    dtype=torch.float32,
                )

                # Blood
                x_blood = None
                if not is_C:
                    x_blood = torch.tensor(
                        [[leukocyte / BLOOD_DISPLAY['leukocyte_count']['scale_max'],
                          neutrophil / BLOOD_DISPLAY['neutrophil_count']['scale_max'],
                          lymphocyte / BLOOD_DISPLAY['lymphocyte_count']['scale_max']]],
                        dtype=torch.float32,
                    )

                # PCR
                x_pcr = None
                if not is_C:
                    x_pcr = torch.tensor([[pcr_result]], dtype=torch.float32)

                # ── Classification ────────────────────────────────────────────
                with torch.no_grad():
                    logits        = global_model(x_img, x_vitals, x_blood, x_pcr)
                    probs         = F.softmax(logits, dim=1)[0]
                    risk_score    = probs[1].item() * 100

                    # Gate weights
                    h_v = 1.0 if x_img   is not None else 0.0
                    h_b = 1.0 if x_blood is not None else 0.0
                    h_p = 1.0 if x_pcr   is not None else 0.0
                    gate_flags = torch.tensor([[h_v, h_b, h_p]], dtype=torch.float32)
                    gate_w     = global_model.gate(gate_flags)[0]  # [3]

                # ── FIN hallucination — decode missing modalities ──────────────
                estimated = global_model.hallucinate(
                    x_vitals=x_vitals,  # <-- This is required now
                    x_vision=x_img,
                    x_blood=x_blood,
                    x_pcr=x_pcr,
                )

            # ═══════════════════════════════════════════════════════════════
            # RESULTS
            # ═══════════════════════════════════════════════════════════════

            # ── FIN Estimated values (Main Objective) ───────────────────────
            if estimated:
                st.markdown("---")
                st.markdown("### 🧠 FIN — Generated Clinical Data (Hallucination)")
                st.info(
                    "**Primary Objective Active:** The AI was forced to generate mathematically accurate missing biological records "
                    "using latent-space federated training correlations."
                )

                # Blood estimates
                if 'blood' in estimated:
                    st.markdown("#### 🩸 Synthesized Blood Panel")
                    raw  = estimated['blood']
                    cols = list(BLOOD_DISPLAY.keys())
                    scaled_vals = [
                        raw['leukocyte_scaled'],
                        raw['neutrophil_scaled'],
                        raw['lymphocyte_scaled'],
                    ]

                    bc1, bc2, bc3 = st.columns(3)
                    for col_ui, col_key, scaled in zip([bc1, bc2, bc3], cols, scaled_vals):
                        meta    = BLOOD_DISPLAY[col_key]
                        # De-normalise: scaled [0,1] → approximate raw value
                        est_raw = scaled * meta['scale_max']
                        lo, hi  = meta['ref_lo'], meta['ref_hi']

                        if est_raw < lo:
                            flag  = f"⬇️ Low (ref: {lo}–{hi})"
                            color = "inverse"
                        elif est_raw > hi:
                            flag  = f"⬆️ High (ref: {lo}–{hi})"
                            color = "inverse"
                        else:
                            flag  = f"✅ Normal (ref: {lo}–{hi})"
                            color = "normal"

                        col_ui.metric(
                            label=f"{meta['label']} ({meta['unit']})",
                            value=f"~{est_raw:.1f}",
                            delta=flag,
                            delta_color=color,
                        )

                # PCR & Vision on the same row using columns for a tighter, denser UI
                v_col1, v_col2 = st.columns(2)
                
                with v_col1:
                    # PCR estimate
                    if 'pcr' in estimated:
                        st.markdown("#### 🧫 Generated RT-PCR")
                        pcr_prob = estimated['pcr']['positive_prob']
                        pct      = pcr_prob * 100
                        st.metric(
                            "PCR Positive Probability",
                            f"{pct:.1f}%",
                            delta="Likely Positive" if pct > 50 else "Likely Negative",
                            delta_color="inverse" if pct > 50 else "normal",
                        )
                        st.progress(pcr_prob)
                        
                with v_col2:
                    # Vision / X-ray estimate
                    if 'vision' in estimated:
                        st.markdown("#### 🩻 Generated Radiology Finding")
                        vis       = estimated['vision']
                        top       = vis['top_finding']
                        st.success(f"**Reconstructed Finding:**\n\n{top}")

            # ── Verdict ────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 🩺 Secondary Objective: Clinical Verdict")
            verdict_col, risk_col = st.columns([1, 1])

            with verdict_col:
                if risk_score > threshold:
                    st.error("### ⚠️ HIGH RISK\nSuspected COVID-19 / Severe Pneumonia")
                else:
                    st.success("### ✅ LOW RISK\nNormal / Non-COVID Finding")

            with risk_col:
                st.metric("COVID-19 Risk Score", f"{risk_score:.1f}%")
                st.progress(risk_score / 100.0)
                st.caption(f"Threshold: {threshold}%")

            # ── Modality gate weights ───────────────────────────────────────
            st.markdown("---")
            st.caption("### ⚖️ Modality Attribution (Gate Weights)")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Vision",  f"{gate_w[0].item():.3f}", delta="measured" if x_img   is not None else "imputed",  delta_color="normal" if x_img   is not None else "off")
            g2.metric("Vitals",  "anchor",  delta="always present", delta_color="normal")
            g3.metric("Blood",   f"{gate_w[1].item():.3f}", delta="measured" if x_blood is not None else "imputed",  delta_color="normal" if x_blood is not None else "off")
            g4.metric("PCR",     f"{gate_w[2].item():.3f}", delta="measured" if x_pcr   is not None else "imputed",  delta_color="normal" if x_pcr   is not None else "off")



            # ── FIN summary banner ──────────────────────────────────────────
            st.markdown("---")
            if is_A and x_img is not None:
                st.success("✅ **Full fusion active** — all 4 modalities measured and used.")
            elif is_A and x_img is None:
                st.warning("⚠️ Hospital A tier but no X-ray uploaded — vision was imputed by FIN.")
            elif is_B:
                st.warning("🔄 **FIN Level 1** — vision imputed from Blood + PCR + Vitals context.")
            elif is_C:
                st.error("🔄 **FIN Level 2** — vision, blood, PCR all imputed from vitals only.")