import pandas as pd
import matplotlib.pyplot as plt
import os
BASE_DIR = "experiments/results"

baseline = pd.read_csv(os.path.join(BASE_DIR, "baseline.csv"))
robust = pd.read_csv(os.path.join(BASE_DIR, "robust_median.csv"))

plt.figure()
plt.plot(baseline["round"], baseline["avg_loss"], label="FedAvg (Baseline)")
plt.plot(robust["round"], robust["avg_loss"], label="Robust Median")

plt.xlabel("Round")
plt.ylabel("Average Loss")
plt.title("Convergence Comparison")
plt.legend()
plt.grid(True)

plt.savefig("experiments/plots/convergence.png")
plt.show()
