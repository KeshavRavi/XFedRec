import os
import yaml
import pandas as pd
from experiments.run_experiment import run_experiment


CONFIGS = {
    "baseline": "experiments/scripts/baseline.yaml",
    "robust_median": "experiments/scripts/robust_median.yaml",
    "byzantine_attack": "experiments/scripts/byzantine_attack.yaml",
    "drift_only": "experiments/scripts/drift_only.yaml",
    "full_system": "experiments/scripts/full_system.yaml",
}

RESULT_DIR = "experiments/results"
os.makedirs(RESULT_DIR, exist_ok=True)


def main():
    for name, cfg_path in CONFIGS.items():
        print(f"\n========== Running {name} ==========")

        metrics = run_experiment(
            config_path=cfg_path,
            return_server=False
        )

        df = pd.DataFrame(metrics)
        out_path = os.path.join(RESULT_DIR, f"{name}.csv")
        df.to_csv(out_path, index=False)

        print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
