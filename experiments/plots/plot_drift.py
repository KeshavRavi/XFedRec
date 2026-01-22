import pandas as pd
import matplotlib.pyplot as plt

import os
BASE_DIR = "experiments/results"

no_drift = pd.read_csv(os.path.join(BASE_DIR, "baseline.csv"))
drift = pd.read_csv(os.path.join(BASE_DIR, "drift_only.csv"))

plt.figure()
plt.plot(no_drift["round"], no_drift["avg_loss"], label="No Drift")
plt.plot(drift["round"], drift["avg_loss"], label="With Concept Drift")

plt.xlabel("Round")
plt.ylabel("Average Loss")
plt.title("Concept Drift Impact")
plt.legend()
plt.grid(True)

plt.savefig("experiments/plots/drift.png")
plt.show()
