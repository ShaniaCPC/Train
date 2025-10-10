import os
import torch
import flwr as fl
from flwr.common import parameters_to_ndarrays
from ultralytics import YOLO

# --- Strategy that remembers the last aggregated parameters ---
class FedAvgWithSave(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            self.final_parameters = aggregated
        return aggregated, metrics


def save_final_model(params, base_ckpt="model/my_model.pt", out_path="static/output/final_model.pt"):
    print("[SERVER] ✅ Aggregation complete, saving model...")

<<<<<<< HEAD
    base_model = YOLO(base_ckpt)
    base_sd = base_model.model.state_dict()
=======
    try:
        # Try to load the base model with safe loading
        base_model = YOLO(base_ckpt)
        base_sd = base_model.model.state_dict()
    except Exception as e:
        print(f"[SERVER] Error loading base model: {e}")
        print("[SERVER] Creating fallback model...")
        # Create a fallback model if loading fails
        base_model = YOLO('yolov8n.pt')
        base_sd = base_model.model.state_dict()
>>>>>>> 1db54ddc3a69759c57047e2b2ce81bd1eb7e9893

    ndarrays = parameters_to_ndarrays(params)
    if len(ndarrays) != len(base_sd):
        raise RuntimeError(
            f"Mismatched param counts: agg={len(ndarrays)} vs model={len(base_sd)}. "
            "Ensure client get_parameters() iterates state_dict in a stable order."
        )

    new_sd = {}
    for (k, v), arr in zip(base_sd.items(), ndarrays):
        t = torch.from_numpy(arr).to(v.device).to(v.dtype)
        if t.shape != v.shape:
            raise RuntimeError(f"Shape mismatch for {k}: got {t.shape}, expected {v.shape}")
        new_sd[k] = t

    base_model.model.load_state_dict(new_sd, strict=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base_model.save(out_path)
    print(f"[SERVER] ✅ Final model saved to: {out_path}")


if __name__ == "__main__":
    strategy = FedAvgWithSave(
<<<<<<< HEAD
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )

    # 4 rounds = (optionally) 4 sequential clients
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=4),
=======
        min_fit_clients=4,  # Require all 4 clients to participate
        min_evaluate_clients=4,  # All clients must evaluate
        min_available_clients=4,  # Wait for all 4 clients to be available
    )

    # Run fewer rounds since all clients train together in each round
    fl.server.start_server(
        server_address="localhost:8082",  # Use port 8082 to avoid conflicts
        config=fl.server.ServerConfig(num_rounds=3),  # 3 rounds with all 4 clients
>>>>>>> 1db54ddc3a69759c57047e2b2ce81bd1eb7e9893
        strategy=strategy,
    )

    if strategy.final_parameters is not None:
        save_final_model(strategy.final_parameters)
    else:
        print("[SERVER] ⚠️ No final parameters were found to save.")
