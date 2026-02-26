import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_ablation(csv_files, labels, title, output_name):
    plt.figure(figsize=(10, 6))
    
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e'] # Red, Blue, Green, Orange
    markers = ['x', 'o', '^', 's']
    linestyles = ['--', '-', '-.', ':']

    for i, (file, label) in enumerate(zip(csv_files, labels)):
        if not os.path.exists(file):
            print(f"⚠️ Warning: Cannot find {file}. Did you run the experiment?")
            continue
            
        df = pd.read_csv(file)
        
        # Plot HR@10 (or avg_loss if you prefer)
        plt.plot(df['round'], df['avg_hr'], 
                 color=colors[i % len(colors)], 
                 marker=markers[i % len(markers)],
                 linestyle=linestyles[i % len(linestyles)],
                 linewidth=2, 
                 label=label)

    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Test Accuracy (HR@10)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    out_path = f"experiments/plots/{output_name}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"✅ Ablation plot saved to {out_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', required=True, help='List of CSV files to plot')
    parser.add_argument('--labels', nargs='+', required=True, help='List of labels for the legend')
    parser.add_argument('--title', type=str, required=True, help='Title of the plot')
    parser.add_argument('--out', type=str, default='ablation_plot', help='Output filename')
    args = parser.parse_args()
    
    plot_ablation(args.files, args.labels, args.title, args.out)