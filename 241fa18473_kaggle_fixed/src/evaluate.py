import os, joblib, numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

def run_evaluation():
    X=np.load("data/processed/X_test_final.npy"); y=np.load("data/processed/y_test.npy"); model=joblib.load("models/random_forest_baseline.pkl")
    pred=model.predict(X); prob=model.predict_proba(X)[:,1]
    print("\n--- Loan Model Evaluation ---")
    print(f"Accuracy : {accuracy_score(y,pred):.4f}"); print(f"Precision: {precision_score(y,pred,zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y,pred,zero_division=0):.4f}"); print(f"F1-Score : {f1_score(y,pred,zero_division=0):.4f}"); print(f"ROC-AUC  : {roc_auc_score(y,prob):.4f}")
if __name__ == "__main__": run_evaluation()
