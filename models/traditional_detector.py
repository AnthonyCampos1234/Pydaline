from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

class TraditionalAnomalyDetector:
    def __init__(self, contamination=0.1):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )

    def fit(self, X):
        if len(X.shape) == 3:
            X = X[:, -1, :]  
        
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        return self

    def predict(self, X):
        if len(X.shape) == 3:
            X = X[:, -1, :] 
        
        X_scaled = self.scaler.transform(X)
        predictions = self.isolation_forest.predict(X_scaled)
        return np.where(predictions == -1, 1, 0)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        scores = self.isolation_forest.score_samples(X_scaled)
        proba = 1 - (scores - scores.min()) / (scores.max() - scores.min())
        return proba 