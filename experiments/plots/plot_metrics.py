import pandas as pd
import matplotlib.pyplot as plt
import os

RESULT_FILE = "experiments/results/drift_only.csv"
OUT_DIR = "experiments/plots"

def plot_metrics():
    if not os.path.exists(RESULT_FILE):
        print("CSV not found.")
        return

    df = pd.read_csv(RESULT_FILE)
    
    # 1. Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df['round'], df['avg_loss'], marker='o', label='MSE Loss', color='blue')
    plt.axvline(x=3, color='black', linestyle='--', label='Drift Start')
    plt.xlabel('Round')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss under Concept Drift')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "metric_loss.png"))
    plt.close()

    # 2. Ranking Metrics Plot (HR & NDCG)
    plt.figure(figsize=(8, 5))
    plt.plot(df['round'], df['avg_hr'], marker='s', label='HR@10', color='green')
    plt.plot(df['round'], df['avg_ndcg'], marker='^', label='NDCG@10', color='orange')
    plt.axvline(x=3, color='black', linestyle='--', label='Drift Start')
    plt.xlabel('Round')
    plt.ylabel('Score')
    plt.title('Ranking Performance (HR / NDCG)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "metric_ranking.png"))
    plt.close()
    
    print(f"✅ Plots saved to {OUT_DIR}")

if __name__ == "__main__":
    plot_metrics()