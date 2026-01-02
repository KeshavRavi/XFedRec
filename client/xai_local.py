# client/xai_local.py

import numpy as np
import os
import pandas as pd
from xai.lime_shap_wrapper import XAIExplainer
from xai.explain_utils import save_text_explanation, save_bar_plot

class LocalXAI:
    def __init__(self, model, feature_names, background_data, device="cpu", out_dir="experiments/results/xai"):
        """
        model: PyTorch model
        feature_names: list, e.g. ['user_id','item_id']
        background_data: DataFrame or np.array for LIME/SHAP
        """
        # Convert np.array to DataFrame if needed
        if isinstance(background_data, np.ndarray):
            background_df = pd.DataFrame(background_data, columns=feature_names)
        else:
            background_df = background_data

        self.wrapper = XAIExplainer(model, background_df, device)
        self.out_dir = out_dir

    def explain_instance(self, background_data, instance, client_id, round_id):
        """
        Generate both LIME and SHAP explanations and save JSON + PNG
        """
        # LIME
        lime_exp = self.wrapper.explain_with_lime(*instance)  # unpack [user_id, item_id]
        lime_weights = dict(lime_exp)

        # SHAP
        shap_values = self.wrapper.explain_with_shap(*instance)

        # Save JSON
        save_text_explanation(
            {"lime": lime_weights, "shap": shap_values},
            self.out_dir,
            f"client_{client_id}_round_{round_id}_explanation.json"
        )

        # Save plot
        save_bar_plot(
            list(lime_weights.keys()),
            list(lime_weights.values()),
            title="LIME Feature Contributions",
            out_path=os.path.join(
                self.out_dir,
                f"client_{client_id}_round_{round_id}_lime.png"
            )
        )
