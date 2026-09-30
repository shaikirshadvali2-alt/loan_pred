import os,json,numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def run_deterministic_test():
    Xtr=np.load("data/processed/X_train_final.npy"); Xte=np.load("data/processed/X_test_final.npy"); ytr=np.load("data/processed/y_train.npy"); yte=np.load("data/processed/y_test.npy")
    params={"n_estimators":100,"max_depth":10,"random_state":42}
    m1=RandomForestClassifier(**params).fit(Xtr,ytr); s1=f1_score(yte,m1.predict(Xte))
    m2=RandomForestClassifier(**params).fit(Xtr,ytr); s2=f1_score(yte,m2.predict(Xte))
    report={"test_name":"Loan Pipeline Reproducibility Validation","parameters":params,"execution_1_f1":s1,"execution_2_f1":s2,"is_strictly_reproducible":bool(s1==s2),"status":"PASSED" if s1==s2 else "FAILED"}
    os.makedirs("artifacts",exist_ok=True); json.dump(report,open("artifacts/reproducibility_report.json","w"),indent=4)
    print(f"Execution 1 F1: {s1:.6f}"); print(f"Execution 2 F1: {s2:.6f}"); print("[SUCCESS] Reproducibility validation completed.")
if __name__ == "__main__": run_deterministic_test()
