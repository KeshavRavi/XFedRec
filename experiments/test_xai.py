from experiments.run_experiment import run_experiment
from xai.lime_shap_wrapper import XAIExplainer
from client.xai_local import LocalXAI
import numpy as np

# run training and get server
server = run_experiment(
    config_path="experiments/scripts/experiment_config.yaml",
    return_server=True
)

# pick ONE trained client
client = server.clients[0]

# pick ONE (user, item) from its local data
row = client.local_data.iloc[0]
user_id = int(row["user_id"])
item_id = int(row["item_id"])

# Vanilla XAIExplainer for printing
explainer = XAIExplainer(
    model=client.model,
    local_df=client.local_data,
    device=client.device
)

print("\n--- LIME EXPLANATION ---")
lime_exp = explainer.explain_with_lime(user_id, item_id)
for feat, val in lime_exp:
    print(f"{feat}: {val:.4f}")

print("\n--- SHAP EXPLANATION ---")
shap_exp = explainer.explain_with_shap(user_id, item_id)
for feat, val in shap_exp.items():
    print(f"{feat}: {val:.4f}")

# ---- LocalXAI for saving JSON + PNG ----
feature_names = ["user_id", "item_id"]

local_xai = LocalXAI(
    model=client.model,
    feature_names=feature_names,
    background_data=client.local_data[['user_id','item_id']]  # THIS DEFINES background_data
)

# Explain and save plots
local_xai.explain_instance(
    background_data=None,  # Not needed anymore inside explain_instance
    instance=np.array([user_id, item_id]),
    client_id=client.client_id,
    round_id=0
)
