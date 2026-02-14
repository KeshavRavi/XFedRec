import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D  # <--- Correct import
import ast
import os

# --- CONFIG ---
RESULT_FILE = "experiments/results/drift_only.csv"
OUTPUT_IMG = "experiments/plots/drift_lifecycle_gantt_fixed.png"
DRIFT_INJECTION_ROUND = 3 

def parse_drift_metrics(row):
    if pd.isna(row): return []
    try:
        return ast.literal_eval(row)
    except:
        return []

def plot_lifecycle():
    if not os.path.exists(RESULT_FILE):
        print(f" File not found: {RESULT_FILE}")
        return

    df = pd.read_csv(RESULT_FILE)
    
    # Store all events per client
    client_events = {} 

    for _, row in df.iterrows():
        round_id = row['round']
        metrics = parse_drift_metrics(row.get('drift_metrics', "[]"))
        
        for m in metrics:
            cid = m['client_id']
            if cid not in client_events: 
                client_events[cid] = {'detect': [], 'recover': []}
            
            if m['metric'] == 'detection_delay':
                client_events[cid]['detect'].append(round_id)
            elif m['metric'] == 'recovery_event':
                client_events[cid]['recover'].append(round_id)

    # Setup Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axvline(x=DRIFT_INJECTION_ROUND, color='black', linestyle='--', linewidth=2, label='True Drift Start')

    y_positions = []
    y_labels = []
    
    sorted_cids = sorted(client_events.keys())
    
    for i, cid in enumerate(sorted_cids):
        events = client_events[cid]
        detections = sorted(events['detect'])
        recoveries = sorted(events['recover'])
        
        # LOGIC FIX: Find the FIRST valid detection after injection
        valid_detect = next((d for d in detections if d >= DRIFT_INJECTION_ROUND), None)
        
        if valid_detect:
            # Find the FIRST recovery that happens AFTER detection
            valid_recover = next((r for r in recoveries if r > valid_detect), None)
            
            y_pos = i * 10
            y_positions.append(y_pos)
            y_labels.append(f"Client {cid}")

            # 1. Draw LAG (Grey): Drift Start -> Detection
            lag_width = valid_detect - DRIFT_INJECTION_ROUND
            if lag_width > 0:
                ax.broken_barh(
                    [(DRIFT_INJECTION_ROUND, lag_width)], 
                    (y_pos - 2, 4), 
                    facecolors='lightgrey', alpha=0.8, hatch='///'
                )

            # 2. Mark Detection (Red X)
            ax.scatter(valid_detect, y_pos, color='red', s=120, zorder=5, marker='X')

            # 3. Draw ADAPTATION (Orange): Detection -> Recovery
            if valid_recover:
                adapt_duration = valid_recover - valid_detect
                ax.broken_barh(
                    [(valid_detect, adapt_duration)], 
                    (y_pos - 2, 4), 
                    facecolors='#ff7f0e', alpha=0.9
                )
                # 4. Mark Recovery (Green Dot)
                ax.scatter(valid_recover, y_pos, color='green', s=120, zorder=5, marker='o')
            else:
                # Not recovered yet (extend to end of chart)
                end_round = df['round'].max()
                ax.broken_barh(
                    [(valid_detect, end_round - valid_detect)], 
                    (y_pos - 2, 4), 
                    facecolors='red', alpha=0.3
                )

    # Custom Legend
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='--', label='True Drift Start'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='red', label='Drift Detected', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Recovered', markersize=10),
        mpatches.Rectangle((0,0),1,1, facecolor='lightgrey', hatch='///', alpha=0.8, label='Lag (Delay)'),
        mpatches.Rectangle((0,0),1,1, facecolor='#ff7f0e', alpha=0.9, label='Adaptation Phase')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Clients")
    ax.set_title("Drift Adaptation Lifecycle: First Incident Analysis")
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f" Corrected plot saved to {OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    plot_lifecycle()