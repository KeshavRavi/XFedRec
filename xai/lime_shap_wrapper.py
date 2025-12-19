# xai/lime_shap_wrapper.py
"""
Unified wrapper for LIME and SHAP explanations.
"""

import numpy as np
import torch
import shap
from lime.lime_tabular import LimeTabularExplainer

class XAIWrapper:
    def __init__(self, model, feature_names, device="cpu"):
        self.model = model
        self.feature_names = feature_names
        self.device = device

    def _predict_fn(self, x):
        """
        Prediction function for LIME/SHAP.
        """
        x = torch.tensor(x, dtype=torch.long).to(self.device)
        user_ids = x[:, 0]
        item_ids = x[:, 1]
        with torch.no_grad():
            preds = self.model(user_ids, item_ids)
        return preds.cpu().numpy()

    def explain_with_lime(self, background_data, instance, num_features=2):
        """
        Generate LIME explanation for a single instance.
        """
        explainer = LimeTabularExplainer(
            background_data,
            feature_names=self.feature_names,
            mode="regression",
            discretize_continuous=False
        )
        exp = explainer.explain_instance(
            instance,
            self._predict_fn,
            num_features=num_features
        )
        return exp

    def explain_with_shap(self, background_data, instance):
        """
        Generate SHAP explanation using KernelExplainer.
        """
        explainer = shap.KernelExplainer(
            self._predict_fn,
            background_data
        )
        shap_values = explainer.shap_values(instance)
        return shap_values
