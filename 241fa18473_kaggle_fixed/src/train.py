import os, joblib, numpy as np
from sklearn.ensemble import RandomForestClassifier

def run_training():
    X=np.load("data/processed/X_train_final.npy"); y=np.load("data/processed/y_train.npy")
    model=RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)
    model.fit(X,y); os.makedirs("models",exist_ok=True); joblib.dump(model,"models/random_forest_baseline.pkl")
    print("[SUCCESS] Random Forest model trained and saved.")
if __name__ == "__main__": run_training()
