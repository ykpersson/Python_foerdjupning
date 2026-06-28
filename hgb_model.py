# hgb_model.py
from sklearn.ensemble import HistGradientBoostingClassifier


class HGBModel:
    def __init__(self, ticker="GROUP_HGB", random_state=535):
        self.ticker = ticker
        self.random_state = random_state
        self.model = None
        self.best_params = None

    # -----------------------------
    # Create model (default or tuned)
    # -----------------------------
    def create_model(self, **params):
        self.model = HistGradientBoostingClassifier(
            random_state=self.random_state,
            **params
        )
        return self.model

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

