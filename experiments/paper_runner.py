import os
import yaml
import pandas as pd
from experiments.run_experiment import run_experiment

# Map the Paper Table Name -> Config File
CONFIGS = {
    "Baseline": "experiments/scripts/paper_baseline.yaml",
    "Robust_FL": "experiments/scripts/paper_robust.yaml",
    "Drift_Aware": "experiments/scripts/paper_drift.yaml",
    "XFedRec_Full": "experiments/scripts/paper_xfedrec.yaml",
}

RESULT_DIR = "experiments/results/paper"
os.makedirs(RESULT_DIR, exist_ok=True)

def main():
    summary_data = []

    for name, cfg_path in CONFIGS.items():
        print(f"\n========== Running {name} ==========")
        
        # Run experiment
        metrics = run_experiment(config_path=cfg_path, return_server=False)
        
        # Save detailed CSV for plotting
        df = pd.DataFrame(metrics)
        out_path = os.path.join(RESULT_DIR, f"{name}.csv")
        df.to_csv(out_path, index=False)
        print(f" Saved results to {out_path}")

        # Collect summary stats for Table 1
        final_loss = df["avg_loss"].iloc[-1]
        final_hr = df["avg_hr"].iloc[-1] if "avg_hr" in df else 0.0
        summary_data.append({
            "Model": name,
            "Final Loss": final_loss,
            "Final HR@10": final_hr
        })

    # Print the Table for your LaTeX
    print("\n\n========== PAPER TABLE DATA ==========")
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)
    summary_df.to_csv(os.path.join(RESULT_DIR, "final_summary_table.csv"), index=False)

if __name__ == "__main__":
    main()