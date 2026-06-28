


class RFModel:
    def __init__(self, ticker="GROUP_RF", random_state=535):
        self.ticker = ticker
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def create_model(self, **params):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            random_state=self.random_state,
            **params
        )
        return self.model

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # RF använder inte valideringsdata – men måste acceptera argumenten
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

