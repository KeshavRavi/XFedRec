# xai/lime_shap_wrapper.py
import shap
from lime import lime_tabular
import numpy as np

class XAIExplainer:
    """
    Wrapper class for LIME and SHAP explanations.
    Works for a client-side recommendation model.
    """

    def __init__(self, model, local_df, device='cpu'):
        self.model = model
        self.local_df = local_df
        self.device = device
        # Example: convert user/item to numpy matrix for explainer
        self.X = self.local_df[['user_id','item_id']].values

    def predict(self, X):
        """
        Accepts 2D numpy array [[user_id, item_id], ...] and returns predictions.
        """
        import torch
        self.model.eval()
        preds = []
        for row in X:
            u = torch.tensor([int(row[0])], dtype=torch.long).to(self.device)
            i = torch.tensor([int(row[1])], dtype=torch.long).to(self.device)
            with torch.no_grad():
                pred = self.model(u, i)
            preds.append(pred.item())
        return np.array(preds)

    def explain_with_lime(self, user_id, item_id):
        explainer = lime_tabular.LimeTabularExplainer(
            self.X,
            feature_names=['user_id', 'item_id'],
            verbose=False,
            mode='regression'
        )
        instance = np.array([user_id, item_id], dtype=float)
        exp = explainer.explain_instance(instance, self.predict)
        return exp.as_list()

    def explain_with_shap(self, user_id, item_id, nsamples=100):
        # take a small sample of background to speed up
        background = self.X[np.random.choice(self.X.shape[0], min(nsamples, self.X.shape[0]), replace=False)]
        explainer = shap.KernelExplainer(self.predict, background)
        instance = np.array([[user_id, item_id]])
        shap_values = explainer.shap_values(instance)
        return dict(zip(['user_id','item_id'], shap_values[0]))
