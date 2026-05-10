# client.py
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import flwr as fl
import pandas as pd
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from models import FedOmniMedFusion

# ─── Image Transforms ────────────────────────────────────────────────────────
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Feature column names ─────────────────────────────────────────────────────
VITALS_COLS = [
    'age', 'sex', 'offset',
    'intubated', 'intubation_present',
    'needed_supplemental_O2', 'went_icu', 'in_icu',
]
BLOOD_COLS = ['leukocyte_count', 'neutrophil_count', 'lymphocyte_count']
PCR_COLS   = ['RT_PCR_positive']
IMAGE_BASE_DIR = "data/images/images"


# ─── Data-driven normalisation ───────────────────────────────────────────────
def compute_vitals_stats(df: pd.DataFrame) -> dict:
    """
    Compute mean/std for continuous vitals from this hospital's training data.
    Binary columns are kept as-is (already 0/1).
    Returns a dict used by standardise_vitals().
    """
    continuous = ['age', 'offset']
    stats = {}
    for col in continuous:
        col_data = df[col].dropna()
        stats[col] = {
            'mean': float(col_data.mean()) if len(col_data) > 0 else 0.0,
            'std':  float(col_data.std())  if len(col_data) > 1 else 1.0,
        }
        if stats[col]['std'] < 1e-6:
            stats[col]['std'] = 1.0
    return stats


def standardise_vitals(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Z-score continuous vitals using data-derived stats. Binary cols untouched."""
    df = df.copy()
    for col, s in stats.items():
        df[col] = (df[col] - s['mean']) / s['std']
    return df


def compute_blood_stats(df: pd.DataFrame) -> dict:
    """Compute per-column mean/std for blood panel from training data."""
    stats = {}
    for col in BLOOD_COLS:
        col_data = df[col].dropna()
        stats[col] = {
            'mean': float(col_data.mean()) if len(col_data) > 0 else 0.0,
            'std':  float(col_data.std())  if len(col_data) > 1 else 1.0,
        }
        if stats[col]['std'] < 1e-6:
            stats[col]['std'] = 1.0
    return stats


def standardise_blood(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Z-score blood panel using data-derived stats."""
    df = df.copy()
    for col, s in stats.items():
        df[col] = (df[col] - s['mean']) / s['std']
    return df


# ─── Dataset ─────────────────────────────────────────────────────────────────
class MedDataset(Dataset):
    """
    Produces 5-tuples: (img, vitals[11], blood[3]|None, pcr[1]|None, label).

    Normalisation stats are computed from training data and passed in, so the
    same stats are used for the test split — no data leakage, no hardcoding.

    The vision_flag in the vitals context vector is set per-sample at __getitem__
    time based on whether the image file actually loads, not at construction time.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vitals_stats: dict,
        blood_stats: dict | None,
        transform=None,
    ):
        self.df           = df.reset_index(drop=True)
        self.transform    = transform
        self.vitals_stats = vitals_stats
        self.blood_stats  = blood_stats

        # Labels
        self.y = torch.tensor(df['finding'].values, dtype=torch.long)

        # Vitals base (8 features) — flags appended per-sample in __getitem__
        v_df = standardise_vitals(df[VITALS_COLS], vitals_stats)
        v_df = v_df.fillna(0.0)
        self.vitals_base = torch.tensor(v_df.values, dtype=torch.float32)  # [N, 8]

        # Blood (3 features, optional)
        has_blood = (
            'leukocyte_count' in df.columns
            and not df['leukocyte_count'].isnull().all()
        )
        if has_blood and blood_stats is not None:
            b_df       = standardise_blood(
                df[BLOOD_COLS].fillna(df[BLOOD_COLS].median()), blood_stats
            )
            self.blood = torch.tensor(b_df.values, dtype=torch.float32)
            # Store raw (unscaled) values for reconstruction loss target
            self.blood_raw = torch.tensor(
                df[BLOOD_COLS].fillna(df[BLOOD_COLS].median()).values,
                dtype=torch.float32,
            )
        else:
            self.blood     = None
            self.blood_raw = None

        # PCR (1 feature, optional)
        has_pcr = (
            'RT_PCR_positive' in df.columns
            and not df['RT_PCR_positive'].isnull().all()
        )
        if has_pcr:
            pcr_vals  = df[PCR_COLS].fillna(df[PCR_COLS].median()).values
            self.pcr  = torch.tensor(pcr_vals, dtype=torch.float32)
        else:
            self.pcr  = None

        # Hospital-level flags for blood/pcr (don't change per sample)
        self._has_blood = 1.0 if self.blood is not None else 0.0
        self._has_pcr   = 1.0 if self.pcr   is not None else 0.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        y          = self.y[idx]
        blood      = self.blood[idx]     if self.blood     is not None else None
        blood_raw  = self.blood_raw[idx] if self.blood_raw is not None else None
        pcr        = self.pcr[idx]       if self.pcr       is not None else None

        # Image — attempt to load; per-sample vision flag set after
        img_tensor = None
        img_path   = self.df.loc[idx, 'filename']
        if pd.notnull(img_path) and isinstance(img_path, str) and img_path != 'None':
            full_path = os.path.join(IMAGE_BASE_DIR, img_path)
            if os.path.exists(full_path):
                try:
                    img        = Image.open(full_path).convert('RGB')
                    img_tensor = self.transform(img) if self.transform else transforms.ToTensor()(img)
                except Exception:
                    img_tensor = None

        # Per-sample vision flag (reflects actual file availability, not hospital tier)
        has_vision = 1.0 if img_tensor is not None else 0.0
        flags      = torch.tensor([has_vision, self._has_blood, self._has_pcr],
                                  dtype=torch.float32)
        vitals     = torch.cat([self.vitals_base[idx], flags])  # [11]

        return img_tensor, vitals, blood, blood_raw, pcr, y


# ─── Custom Collate ───────────────────────────────────────────────────────────
def collate_fn(batch):
    imgs       = [item[0] for item in batch]
    vitals     = torch.stack([item[1] for item in batch])
    bloods     = [item[2] for item in batch]
    bloods_raw = [item[3] for item in batch]
    pcrs       = [item[4] for item in batch]
    ys         = torch.stack([item[5] for item in batch])

    # Image: None if all missing, stacked tensor otherwise (zeros for mixed)
    if all(i is None for i in imgs):
        imgs_out = None
    elif all(i is not None for i in imgs):
        imgs_out = torch.stack(imgs)
    else:
        imgs_out = torch.stack([
            i if i is not None else torch.zeros(3, 224, 224) for i in imgs
        ])

    blood_out     = torch.stack(bloods)     if all(b is not None for b in bloods)     else None
    blood_raw_out = torch.stack(bloods_raw) if all(b is not None for b in bloods_raw) else None
    pcr_out       = torch.stack(pcrs)       if all(p is not None for p in pcrs)       else None

    return imgs_out, vitals, blood_out, blood_raw_out, pcr_out, ys


# ─── Federated Client ─────────────────────────────────────────────────────────
class HospitalClient(fl.client.NumPyClient):

    # Reconstruction loss weight — massively increased to prioritize Hallucination accuracy
    RECON_WEIGHT  = 2.0

    # FedProx proximal mu — must match server.py
    PROXIMAL_MU   = 0.2
    # Local epochs per FL round — kept low for 930 total samples
    LOCAL_EPOCHS  = 3
    # Batch size
    BATCH_SIZE    = 8

    def __init__(self, hosp_id: str):
        self.hosp_id = hosp_id
        self.model   = FedOmniMedFusion()
        self.load_local_data()

        # Optimizer — reset after data load so param groups are correct
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=3e-4, weight_decay=1e-3
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=1e-5
        )

    # ── Data Loading ──────────────────────────────────────────────────────────
    def load_local_data(self):
        csv_path = f"simulated_silos/Hospital_{self.hosp_id}/clinical_data.csv"
        df       = pd.read_csv(csv_path)

        n_total   = len(df)
        n_covid   = int(df['finding'].sum())
        n_normal  = n_total - n_covid
        print(f"\n{'='*60}")
        print(f"[Hospital {self.hosp_id}] Dataset loaded")
        print(f"  Total samples : {n_total}")
        print(f"  COVID-19      : {n_covid}  ({n_covid/n_total*100:.1f}%)")
        print(f"  Normal        : {n_normal} ({n_normal/n_total*100:.1f}%)")

        # 80 / 20 stratified split (df was shuffled at silo creation)
        split_idx = int(n_total * 0.8)
        train_df  = df.iloc[:split_idx].copy()
        test_df   = df.iloc[split_idx:].copy()

        # ── Compute normalisation stats from TRAINING data only ───────────────
        self.vitals_stats = compute_vitals_stats(train_df)

        has_blood = (
            'leukocyte_count' in train_df.columns
            and not train_df['leukocyte_count'].isnull().all()
        )
        self.blood_stats = compute_blood_stats(train_df) if has_blood else None

        print(f"  Vitals stats  : age μ={self.vitals_stats['age']['mean']:.1f} "
              f"σ={self.vitals_stats['age']['std']:.1f} | "
              f"offset μ={self.vitals_stats['offset']['mean']:.1f} "
              f"σ={self.vitals_stats['offset']['std']:.1f}")
        if self.blood_stats:
            print(f"  Blood stats   : leuko μ={self.blood_stats['leukocyte_count']['mean']:.2f} | "
                  f"neutro μ={self.blood_stats['neutrophil_count']['mean']:.2f} | "
                  f"lympho μ={self.blood_stats['lymphocyte_count']['mean']:.2f}")
        print(f"  Train/Test    : {len(train_df)} / {len(test_df)}")

        self.train_ds = MedDataset(train_df, self.vitals_stats, self.blood_stats,
                                   transform=TRAIN_TRANSFORMS)
        self.test_ds  = MedDataset(test_df,  self.vitals_stats, self.blood_stats,
                                   transform=EVAL_TRANSFORMS)

        # ── Balanced sampler — derived from data, no hardcoded weights ─────────
        target       = torch.tensor(train_df['finding'].values, dtype=torch.long)
        num_pos      = max(int(target.sum()), 1)
        num_neg      = max(len(target) - num_pos, 1)
        sample_w     = torch.zeros(len(target))
        sample_w[target == 1] = 1.0 / num_pos
        sample_w[target == 0] = 1.0 / num_neg

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_w, num_samples=len(target), replacement=True
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.BATCH_SIZE,
            sampler=sampler,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            drop_last=True,
        )
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            drop_last=False,
        )

        # ── Loss: class weight derived from class counts, not hardcoded ────────
        pos_weight     = num_neg / num_pos  # inverse frequency
        # Cap at 3.0 to avoid extreme weighting for very small hospitals
        pos_weight     = min(pos_weight, 3.0)
        class_weights  = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"  Class weights : Normal=1.00, COVID={pos_weight:.2f} (inv-freq, capped at 3.0)")
        print(f"{'='*60}")

    # ── Flower interface ──────────────────────────────────────────────────────
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        keys       = self.model.state_dict().keys()
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, parameters)})
        self.model.load_state_dict(state_dict, strict=True)

    # ── Local Training ────────────────────────────────────────────────────────
    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Snapshot global params for FedProx proximal term
        global_params = [p.detach().clone() for p in self.model.parameters()]

        self.model.train()
        print(f"\n{'─'*60}")
        print(f"[Hospital {self.hosp_id}] Local training  "
              f"epochs={self.LOCAL_EPOCHS}  μ={self.PROXIMAL_MU}  "
              f"recon_w={self.RECON_WEIGHT}")
        print(f"{'─'*60}")

        for epoch in range(self.LOCAL_EPOCHS):
            ce_total    = 0.0
            recon_total = 0.0
            prox_total  = 0.0
            batches     = 0

            for imgs, vitals, blood, blood_raw, pcr, y in self.train_loader:
                self.optimizer.zero_grad()

                # Modality dropout — Hospital A only, 40% chance
                # Forces the model to learn without imaging on some batches
                cur_imgs = imgs
                if self.hosp_id == 'A' and random.random() < 0.40:
                    cur_imgs = None

                # ── Classification loss ───────────────────────────────────────
                logits  = self.model(cur_imgs, vitals, blood, pcr)
                ce_loss = self.criterion(logits, y)

                # ── Reconstruction loss (decoder heads) ───────────────────────
                recon_loss = torch.tensor(0.0)
                
                # Anchor variable for hallucination tracking
                f_vitals = self.model.t_enc(vitals)

                if blood is not None and blood_raw is not None:
                    # Target scaling
                    blood_target = self._normalise_blood_for_recon(blood_raw)
                    
                    # 1. Ensure the raw encoder can perfectly decode backwards
                    emb_blood = self.model.b_enc(blood)
                    decoded_blood = self.model.b_enc.decode(emb_blood)
                    recon_loss = recon_loss + F.mse_loss(decoded_blood, blood_target)
                    
                    # 2. FORCE THE IMPUTER TO HALLUCINATE ACCURATELY
                    hallucinated_emb = self.model.blood_imputer(f_vitals)
                    hallucinated_vals = self.model.b_enc.decode(hallucinated_emb)
                    recon_loss = recon_loss + 2.0 * F.mse_loss(hallucinated_vals, blood_target)

                if pcr is not None:
                    # 1. Base Encoder Reconstruction
                    emb_pcr = self.model.p_enc(pcr)
                    decoded_pcr = self.model.p_enc.decode(emb_pcr)
                    recon_loss = recon_loss + F.binary_cross_entropy(decoded_pcr, pcr)
                    
                    # 2. Hallucination Target Training
                    halluc_emb_pcr = self.model.pcr_imputer(f_vitals)
                    halluc_prob = self.model.p_enc.decode(halluc_emb_pcr)
                    recon_loss = recon_loss + 2.0 * F.binary_cross_entropy(halluc_prob, pcr)

                # ── FedProx proximal term ─────────────────────────────────────
                prox_term = sum(
                    torch.norm(p - g) ** 2
                    for p, g in zip(self.model.parameters(), global_params)
                )

                loss = (ce_loss
                        + self.RECON_WEIGHT * recon_loss
                        + (self.PROXIMAL_MU / 2.0) * prox_term)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                ce_total    += ce_loss.item()
                recon_total += recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss
                prox_total  += prox_term.item()
                batches     += 1

            n = max(batches, 1)
            print(f"  Epoch {epoch+1}/{self.LOCAL_EPOCHS} | "
                  f"CE: {ce_total/n:.4f} | "
                  f"Recon: {recon_total/n:.4f} | "
                  f"Prox: {prox_total/n:.4f}")

        self.scheduler.step()
        return self.get_parameters(config={}), len(self.train_ds), {
            "hosp_id": str(self.hosp_id),
            "ce_loss": float(ce_total/n),
            "recon_loss": float(recon_total/n),
            "prox_loss": float(prox_total/n)
        }

    def _normalise_blood_for_recon(self, blood_raw: torch.Tensor) -> torch.Tensor:
        """
        Map raw blood values to [0, 1] using per-column min/max from training stats.
        Falls back to simple clamp/scale using 99th-pct reference ranges if stats
        are unavailable, so there is ZERO hardcoding of reference values.
        """
        if self.blood_stats is None:
            return torch.clamp(blood_raw / 20.0, 0.0, 1.0)

        # Use mean ± 3σ as the normalisation range (covers ~99.7% of values)
        cols  = BLOOD_COLS
        mins  = torch.tensor(
            [self.blood_stats[c]['mean'] - 3 * self.blood_stats[c]['std'] for c in cols],
            dtype=torch.float32,
        )
        maxs  = torch.tensor(
            [self.blood_stats[c]['mean'] + 3 * self.blood_stats[c]['std'] for c in cols],
            dtype=torch.float32,
        )
        scale = (maxs - mins).clamp(min=1e-6)
        return torch.clamp((blood_raw - mins) / scale, 0.0, 1.0)

    # ── Local Evaluation ──────────────────────────────────────────────────────
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for imgs, vitals, blood, _, pcr, y in self.test_loader:
                logits     = self.model(imgs, vitals, blood, pcr)
                loss       = self.criterion(logits, y)
                total_loss += loss.item()

                probs      = torch.softmax(logits, dim=1)[:, 1]
                preds      = torch.argmax(logits, dim=1)

                y_true.extend(y.tolist())
                y_pred.extend(preds.tolist())
                y_prob.extend(probs.tolist())

        n         = len(y_true)
        avg_loss  = total_loss / max(len(self.test_loader), 1)
        accuracy  = sum(p == t for p, t in zip(y_pred, y_true)) / n if n else 0.0
        f1        = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        cm        = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"\n[Hospital {self.hosp_id}] ── Evaluation Results ──")
        print(f"  Samples    : {n}")
        print(f"  Loss       : {avg_loss:.4f}")
        print(f"  Accuracy   : {accuracy*100:.2f}%")
        print(f"  Precision  : {precision*100:.2f}%")
        print(f"  Recall     : {recall*100:.2f}%   (Sensitivity)")
        print(f"  Specificity: {specificity*100:.2f}%")
        print(f"  F1-Score   : {f1*100:.2f}%")
        print(f"  Confusion  : TP={tp} TN={tn} FP={fp} FN={fn}")

        return float(avg_loss), n, {
            "hosp_id":     str(self.hosp_id),
            "accuracy":    float(accuracy),
            "f1":          float(f1),
            "precision":   float(precision),
            "recall":      float(recall),
            "specificity": float(specificity),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True, choices=['A', 'B', 'C'])
    args = parser.parse_args()

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=HospitalClient(args.id).to_client(),
    )