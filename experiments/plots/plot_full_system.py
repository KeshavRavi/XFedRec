import pandas as pd
import matplotlib.pyplot as plt
import os

# Point to the "paper" results folder
RESULTS_DIR = "experiments/results/paper"
OUT_IMG = "experiments/plots/fig3_results.png"

def plot_comparison():
    # Map the labels to your actual filenames
    files = {
        "Baseline": "Baseline.csv",
        "Robust FL": "Robust_FL.csv",
        "Drift-Aware": "Drift_Aware.csv",
        "XFedRec (Proposed)": "XFedRec_Full.csv"
    }
    
    plt.figure(figsize=(10, 6))
    
    # Define colors for consistency
    colors = {"Baseline": "blue", "Robust FL": "orange", "Drift-Aware": "green", "XFedRec (Proposed)": "red"}
    
    for label, filename in files.items():
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Plot Loss
            plt.plot(df["round"], df["avg_loss"], marker='o', linewidth=2, label=label, color=colors.get(label))
        else:
            print(f"⚠️ Warning: Missing {filename}")

    # Mark the event round
    plt.axvline(x=3, color='black', linestyle='--', alpha=0.8, label='Drift/Attack Start')
    
    plt.title("Robustness & Adaptation: Training Loss Comparison")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Training Loss (MSE)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save
    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=300)
    print(f" Comparison plot saved to {OUT_IMG}")

if __name__ == "__main__":
    plot_comparison()