import pandas as pd
import matplotlib.pyplot as plt
import os

FEDAVG_FILE = "experiments/results/attack_fedavg.csv"
ROBUST_FILE = "experiments/results/attack_robust.csv"
OUT_IMG = "experiments/plots/robustness_comparison.png"

def plot_robustness():
    if not os.path.exists(FEDAVG_FILE) or not os.path.exists(ROBUST_FILE):
        print("CSV files not found. Run both experiments first.")
        return

    df_fedavg = pd.read_csv(FEDAVG_FILE)
    df_robust = pd.read_csv(ROBUST_FILE)
    
    plt.figure(figsize=(9, 6))
    
    # Plot FedAvg (Vulnerable)
    plt.plot(df_fedavg['round'], df_fedavg['avg_loss'], 
             marker='x', linestyle='--', color='red', linewidth=2, 
             label='FedAvg (2 Attackers)')
    
    # Plot Median (Robust)
    plt.plot(df_robust['round'], df_robust['avg_loss'], 
             marker='o', linestyle='-', color='blue', linewidth=2, 
             label='Coordinate Median (2 Attackers)')
    
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Test Loss (MSE)', fontsize=12)
    plt.title('System Robustness Under Byzantine Attack (Scaling Attack)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Set y-axis to logarithmic if FedAvg exploded massively
    if df_fedavg['avg_loss'].max() > 50:
        plt.yscale('log')
        plt.ylabel('Test Loss (MSE) - Log Scale', fontsize=12)
        
    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_IMG)
    print(f"✅ Robustness plot saved to {OUT_IMG}")
    plt.show()

if __name__ == "__main__":
    plot_robustness()