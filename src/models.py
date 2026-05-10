import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# ─── Radiological finding labels (vision decoder output) ──────────────────────
VISION_FINDINGS = [
    "Normal / No Finding",
    "Bilateral Ground-Glass Opacity",
    "Unilateral Consolidation",
    "Pleural Effusion",
    "Other Pathology",
]

# ─── Individual Modality Encoders ─────────────────────────────────────────────

class VisionEncoder(nn.Module):
    """
    ResNet-18 backbone with early layers frozen.
    STATIC FALLBACK REMOVED: Missing images are now imputed dynamically in the Fusion block.
    """
    def __init__(self, output_dim: int = 128):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for name, param in backbone.named_parameters():
            if any(layer in name for layer in ('conv1', 'bn1', 'layer1', 'layer2')):
                param.requires_grad = False

        backbone.fc = nn.Sequential(
            nn.Linear(backbone.fc.in_features, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


class VitalsEncoder(nn.Module):
    """Encodes the 11-feature vitals vector (8 clinical + 3 context flags)."""
    def __init__(self, input_dim: int = 11, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class BloodEncoder(nn.Module):
    """
    Encodes 3-feature blood panel (leukocyte, neutrophil, lymphocyte).
    Includes a decoder head that maps the embedding back to scaled blood values.
    """
    def __init__(self, input_dim: int = 3, output_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.ReLU(),
        )
        # Decoder: embedding → estimated scaled blood values [0, 1] range
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid(),   # keeps outputs in [0, 1] matching training normalisation
        )

    def forward(self, x):
        return self.encoder(x)

    def decode(self, embedding):
        """Map an embedding back to estimated scaled values."""
        return self.decoder(embedding)


class PCREncoder(nn.Module):
    """
    Encodes the single RT-PCR flag.
    Decoder maps the embedding back to a PCR-positive probability.
    """
    def __init__(self, input_dim: int = 1, output_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.ReLU(),
        )
        # Decoder: embedding → PCR-positive probability
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.encoder(x)

    def decode(self, embedding):
        return self.decoder(embedding)


# ─── Global Fusion Model ──────────────────────────────────────────────────────

class FedOmniMedFusion(nn.Module):
    """
    Multimodal fusion with:
      - Modality-Aware Gating (MAG): sigmoid gate per missing modality
      - Conditional Imputers: Translates Vitals into missing modalities dynamically
      - Decoder heads for interpretable imputation outputs
    
    Fused dimension: 128 (vision) + 64 (vitals) + 32 (blood) + 32 (pcr) = 256
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.v_enc = VisionEncoder(output_dim=128)
        self.t_enc = VitalsEncoder(input_dim=11, output_dim=64)
        self.b_enc = BloodEncoder(input_dim=3, output_dim=32)
        self.p_enc = PCREncoder(input_dim=1, output_dim=32)

        # ─── CONDITIONAL IMPUTERS ─────────────────────────────────────────────
        # These learn to generate the missing embeddings based on the Vitals (dim 64)
        self.vision_imputer = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.blood_imputer = nn.Linear(64, 32)
        self.pcr_imputer = nn.Linear(64, 32)

        # Vision decoder: embedding → radiological finding probabilities
        # Output: len(VISION_FINDINGS) = 5 classes
        self.vision_decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(VISION_FINDINGS)),
        )

        # Modality gate: independent sigmoid per gated modality (3: vision, blood, pcr)
        self.gate = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid(),
        )

        # Classifier: 256-dim fused representation → num_classes
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def _get_flags(self, x_vision, x_blood, x_pcr, batch_size, device):
        """Build the [B, 3] gate flag tensor from modality presence."""
        has_v = 1.0 if x_vision is not None else 0.0
        has_b = 1.0 if x_blood  is not None else 0.0
        has_p = 1.0 if x_pcr    is not None else 0.0
        return torch.tensor(
            [[has_v, has_b, has_p]] * batch_size,
            device=device, dtype=torch.float32,
        )

    def forward(self, x_vision=None, x_vitals=None, x_blood=None, x_pcr=None):
        batch_size = x_vitals.shape[0]
        device     = x_vitals.device

        # ── Encode Vitals (Always Present Anchor) ─────────────────────────────
        f_vitals = self.t_enc(x_vitals)

        # ── Encode or Impute conditionally based on Vitals ────────────────────
        f_vision = self.v_enc(x_vision) if x_vision is not None else self.vision_imputer(f_vitals)
        f_blood  = self.b_enc(x_blood)  if x_blood is not None  else self.blood_imputer(f_vitals)
        f_pcr    = self.p_enc(x_pcr)    if x_pcr is not None    else self.pcr_imputer(f_vitals)

        # ── Gate weights ──────────────────────────────────────────────────────
        flags = self._get_flags(x_vision, x_blood, x_pcr, batch_size, device)
        w     = self.gate(flags)   # [B, 3] — sigmoid, independent per modality

        # ── Weighted fusion ───────────────────────────────────────────────────
        fused = torch.cat([
            f_vision * w[:, 0:1],
            f_vitals,                # anchor — no gate
            f_blood  * w[:, 1:2],
            f_pcr    * w[:, 2:3],
        ], dim=1)                    # [B, 256]

        return self.classifier(fused)

    # ── Hallucination / imputation decoder ────────────────────────────────────
    @torch.no_grad()
    def hallucinate(self, x_vitals, x_vision=None, x_blood=None, x_pcr=None, batch_size=1, device=None):
        """
        Returns a dict of dynamically estimated clinical values for any MISSING modality.
        """
        if device is None:
            device = next(self.parameters()).device

        result = {}
        
        # Encode the specific patient's vitals to drive the conditional hallucination
        f_vitals = self.t_enc(x_vitals) 

        if x_blood is None:
            emb  = self.blood_imputer(f_vitals)
            vals = self.b_enc.decode(emb)[0]   
            result['blood'] = {
                'leukocyte_scaled':  vals[0].item(),
                'neutrophil_scaled': vals[1].item(),
                'lymphocyte_scaled': vals[2].item(),
            }

        if x_pcr is None:
            emb  = self.pcr_imputer(f_vitals)
            prob = self.p_enc.decode(emb)[0, 0].item()
            result['pcr'] = {'positive_prob': prob}

        if x_vision is None:
            emb    = self.vision_imputer(f_vitals)
            logits = self.vision_decoder(emb)[0]        
            probs  = torch.softmax(logits, dim=0).tolist()
            top_i  = int(torch.argmax(logits).item())
            result['vision'] = {
                'finding_probs':  probs,
                'finding_labels': VISION_FINDINGS,
                'top_finding':    VISION_FINDINGS[top_i],
            }

        return result