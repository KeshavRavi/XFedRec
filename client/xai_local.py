# client/xai_local.py
"""
Client-side explainability module.
"""

import numpy as np
import os
from ..xai.lime_shap_wrapper import XAIWrapper
from ..xai.explain_utils import save_text_explanation, save_bar_plot

class LocalXAI:
    def __init__(self, model, feature_names, device="cpu", out_dir="experiments/results/xai"):
        self.wrapper = XAIWrapper(model, feature_names, device)
        self.out_dir = out_dir

    def explain_instance(self, background_data, instance, client_id, round_id):
        """
        Generate both LIME and SHAP explanations.
        """
        # LIME
        lime_exp = self.wrapper.explain_with_lime(background_data, instance)
        lime_weights = dict(lime_exp.as_list())

        # SHAP
        shap_values = self.wrapper.explain_with_shap(background_data, instance)

        # Save textual explanations
        save_text_explanation(
            {
                "lime": lime_weights,
                "shap": shap_values.tolist()
            },
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
