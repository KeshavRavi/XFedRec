import pandas as pd
import matplotlib.pyplot as plt

import os
BASE_DIR = "experiments/results"

baseline = pd.read_csv(os.path.join(BASE_DIR, "baseline.csv"))
median = pd.read_csv(os.path.join(BASE_DIR, "robust_median.csv"))
drift = pd.read_csv(os.path.join(BASE_DIR, "drift_only.csv"))
full = pd.read_csv(os.path.join(BASE_DIR, "full_system.csv"))

plt.figure()
plt.plot(baseline["round"], baseline["avg_loss"], label="Baseline")
plt.plot(median["round"], median["avg_loss"], label="Robust Aggregation")
plt.plot(drift["round"], drift["avg_loss"], label="Drift-Aware")
plt.plot(full["round"], full["avg_loss"], label="XFedRec (Full System)", linewidth=3)

plt.xlabel("Round")
plt.ylabel("Average Loss")
plt.title("XFedRec: Full System Evaluation")
plt.legend()
plt.grid(True)

plt.savefig("experiments/plots/full_system.png")
plt.show()
