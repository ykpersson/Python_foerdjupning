from catboost import CatBoostClassifier
import numpy as np

class CatBoostModel:
    def __init__(self, ticker="GROUP_CAT", random_state=535):
        self.ticker = ticker
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def create_model(self, **params):

        if params.get("bootstrap_type", None) == "Bayesian":
            params["subsample"] = None

        self.model = CatBoostClassifier(
            random_seed=self.random_state,
            verbose=False,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            od_type="Iter",
            od_wait=30,
            **params
        )
        return self.model

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val)
            )
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X):
        preds = self.model.predict(X)
        # CatBoost returns strings → convert to int
        preds = np.array(preds, dtype=int).flatten()
        return preds

    def predict_proba(self, X):
        return self.model.predict_proba(X)