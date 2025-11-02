import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self # nothing to learn

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')


class FeatureSelectorByName(BaseEstimator, TransformerMixin):
    def __init__(self, keep_names, all_names):
        self.keep_names = list(keep_names)
        self.all_names = list(all_names)   # the names returned by preprocessor.get_feature_names_out()

    def fit(self, X, y=None):
        # compute indices of the kept names (matching heuristics below)
        # try exact match first, then fallback to substring matching for safety
        indices = []
        name_to_index = {name: i for i, name in enumerate(self.all_names)}
        for want in self.keep_names:
            if want in name_to_index:
                indices.append(name_to_index[want])
            else:
                # fallback: find the first feature name that contains the want string (useful when prefixes differ)
                matches = [i for i, nm in enumerate(self.all_names) if want in nm]
                if matches:
                    indices.append(matches[0])
                else:
                    # warn but continue
                    print(f"FeatureSelectorByName warning: '{want}' not found in encoded feature list.")
        self.indices_ = sorted(set(indices))
        return self

    def transform(self, X):
        # X may be array (from ColumnTransformer) — select columns by index
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("FeatureSelectorByName expects 2D array-like input")
        return X[:, self.indices_]

    def get_feature_names_out(self, input_features=None):
        # return actual selected names
        return np.array(self.all_names)[self.indices_]
