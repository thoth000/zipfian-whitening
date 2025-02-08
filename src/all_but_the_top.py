from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from torchtyping import TensorType as TT


class AllButTheTop:
    def __init__(self, components_threshhold: int = 3):
        self.components_threshhold = components_threshhold
        self.pca = PCA(n_components=components_threshhold, random_state=0)
        self.is_fitted = False

    def fit(
        self, W: TT["num_words", "hidden_dim"], p: Optional[TT["num_words"]] = None
    ):
        device = W.device
        # convert to numpy
        W = W.cpu().numpy()
        # centering
        self.mu = W.mean(axis=0)
        W = W - self.mu
        self.mu = torch.tensor(self.mu, device=device)
        self.pca.fit(W)
        self.is_fitted = True

        # dominant components
        self.common_components: TT["D", "hidden_dim"] = self.pca.components_[
            : self.components_threshhold
        ]
        self.common_components = torch.tensor(self.common_components, device=device)
        return self

    def transform(
        self, W: TT["num_words", "hidden_dim"]
    ) -> TT["num_words", "hidden_dim"]:
        if not self.is_fitted:
            raise ValueError("All-but-the-top has not been fitted yet.")
        # convert to numpy & centering
        W = W - self.mu

        # remove word-level the common components (all-but-the-top)
        W = W - (W @ self.common_components.T) @ self.common_components

        return W

    def fit_transform(
        self, W: TT["num_words", "hidden_dim"], p: Optional[TT["num_words"]] = None
    ) -> TT["num_words", "hidden_dim"]:
        self.fit(W, p)
        return self.transform(W)
