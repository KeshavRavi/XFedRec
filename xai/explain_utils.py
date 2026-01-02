import json
import matplotlib.pyplot as plt
import os

def save_text_explanation(explanation_dict, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, 'w') as f:
        json.dump(explanation_dict, f, indent=4)

def save_bar_plot(features, values, title, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.bar(features, values)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
