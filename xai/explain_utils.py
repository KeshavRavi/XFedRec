# xai/explain_utils.py
"""
Utility functions for formatting and saving explanations.
"""

import os
import json
import matplotlib.pyplot as plt

def save_text_explanation(explanation_dict, out_dir, fname):
    """
    Save explanation as JSON.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        json.dump(explanation_dict, f, indent=2)

def save_bar_plot(features, weights, title, out_path):
    """
    Save a bar plot of feature contributions.
    """
    plt.figure()
    plt.barh(features, weights)
    plt.title(title)
    plt.xlabel("Contribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
