# server.py
import flwr as fl
import torch
import os
from collections import OrderedDict
from models import FedOmniMedFusion

SAVE_PATH   = "saved_models/global_model_final.pth"
NUM_ROUNDS  = 30
PROXIMAL_MU = 0.2   # must match client.py HospitalClient.PROXIMAL_MU


class SaveModelFedProx(fl.server.strategy.FedProx):
    """
    FedProx strategy with:
      - Full per-hospital AND global metrics printed to terminal every round
      - Model saved after the final round
    """

    def _print_divider(self, char: str = "─", width: int = 68):
        print(char * width)

    def aggregate_evaluate(self, server_round, results, failures):
        loss_agg, metrics_agg = super().aggregate_evaluate(server_round, results, failures)

        if not results:
            print(f"\n[Round {server_round}] No evaluation results received.")
            return loss_agg, metrics_agg

        # ── Per-hospital breakdown ────────────────────────────────────────────
        print(f"\n{'═'*68}")
        print(f"  ROUND {server_round:>2} / {NUM_ROUNDS}  ──  PER-HOSPITAL RESULTS")
        print(f"{'═'*68}")

        hosp_labels = ['A', 'B', 'C']
        all_metrics = []

        for i, (_, res) in enumerate(results):
            m       = res.metrics
            # Retrieve the true hospital ID sent by the client, fallback to i if missing
            label   = m.get('hosp_id', str(i))
            n       = res.num_examples
            all_metrics.append((n, m))

            acc   = m.get('accuracy',    0) * 100
            f1    = m.get('f1',          0) * 100
            prec  = m.get('precision',   0) * 100
            rec   = m.get('recall',      0) * 100
            spec  = m.get('specificity', 0) * 100
            loss  = res.loss

            print(f"\n  Hospital {label}  ({n} test samples)")
            self._print_divider()
            print(f"  {'Loss':<14}: {loss:.4f}")
            print(f"  {'Accuracy':<14}: {acc:.2f}%")
            print(f"  {'Precision':<14}: {prec:.2f}%")
            print(f"  {'Recall (Sens)':<14}: {rec:.2f}%")
            print(f"  {'Specificity':<14}: {spec:.2f}%")
            print(f"  {'F1-Score':<14}: {f1:.2f}%")

        # ── Weighted global averages ───────────────────────────────────────────
        total_n   = sum(n for n, _ in all_metrics)
        metric_keys = ['accuracy', 'f1', 'precision', 'recall', 'specificity']

        global_metrics = {
            k: sum(m.get(k, 0) * n for n, m in all_metrics) / total_n
            for k in metric_keys
        }

        print(f"\n{'═'*68}")
        print(f"  ROUND {server_round:>2} / {NUM_ROUNDS}  ──  GLOBAL WEIGHTED AVERAGES  ({total_n} samples)")
        print(f"{'═'*68}")
        print(f"  {'Global Loss':<16}: {loss_agg:.4f}")
        print(f"  {'Accuracy':<16}: {global_metrics['accuracy']*100:.2f}%")
        print(f"  {'Precision':<16}: {global_metrics['precision']*100:.2f}%")
        print(f"  {'Recall (Sens)':<16}: {global_metrics['recall']*100:.2f}%")
        print(f"  {'Specificity':<16}: {global_metrics['specificity']*100:.2f}%")
        print(f"  {'F1-Score':<16}: {global_metrics['f1']*100:.2f}%")
        print(f"{'═'*68}\n")

        # Pass global metrics upstream for Flower history
        return loss_agg, {k: global_metrics[k] for k in metric_keys}

    def aggregate_fit(self, server_round, results, failures):
        agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)

        if results:
            print(f"\n{'═'*68}")
            print(f"  ROUND {server_round:>2} / {NUM_ROUNDS}  ──  TRAINING METRICS SUMMARY")
            print(f"{'═'*68}")
            for idx, (_, res) in enumerate(results):
                m = res.metrics
                label = m.get('hosp_id', str(idx))
                ce = m.get('ce_loss', 0.0)
                recon = m.get('recon_loss', 0.0)
                prox = m.get('prox_loss', 0.0)
                print(f"  Hospital {label:<2} | CE Loss: {ce:.4f}  |  Recon Loss: {recon:.4f}  |  Prox: {prox:.4f}")
            print(f"{'═'*68}\n")

        if agg_params is not None and server_round == NUM_ROUNDS:
            print(f"\n[Server] Saving final global model → {SAVE_PATH}")
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

            ndarrays   = fl.common.parameters_to_ndarrays(agg_params)
            model      = FedOmniMedFusion()
            params_zip = zip(model.state_dict().keys(), ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_zip})
            torch.save(state_dict, SAVE_PATH)
            print(f"[Server] Model saved successfully.\n")

        return agg_params, agg_metrics


def main():
    print("=" * 68)
    print("  Fed-OmniMed Server  —  FedProx Strategy")
    print(f"  Rounds: {NUM_ROUNDS}   proximal_mu: {PROXIMAL_MU}")
    print("=" * 68)

    strategy = SaveModelFedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        proximal_mu=PROXIMAL_MU,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()