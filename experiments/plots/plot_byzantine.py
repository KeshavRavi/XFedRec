import pandas as pd
import matplotlib.pyplot as plt
import os
BASE_DIR = "experiments/results"

baseline = pd.read_csv(os.path.join(BASE_DIR, "baseline.csv"))
attack = pd.read_csv(os.path.join(BASE_DIR, "byzantine_attack.csv"))

plt.figure()
plt.plot(baseline["round"], baseline["avg_loss"], label="No Attack")
plt.plot(attack["round"], attack["avg_loss"], label="Byzantine Attack")

plt.xlabel("Round")
plt.ylabel("Average Loss")
plt.title("Robustness Under Byzantine Attack")
plt.legend()
plt.grid(True)

plt.savefig("experiments/plots/byzantine.png")
plt.show()
