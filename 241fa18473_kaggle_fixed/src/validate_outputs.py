import os,json,numpy as np
def validate_preprocessing_outputs():
    Xtr=np.load("data/processed/X_train_final.npy"); Xte=np.load("data/processed/X_test_final.npy"); ytr=np.load("data/processed/y_train.npy"); yte=np.load("data/processed/y_test.npy"); errors=[]
    if np.isnan(Xtr).sum(): errors.append("NaNs detected in X_train")
    if np.isnan(Xte).sum(): errors.append("NaNs detected in X_test")
    if Xtr.shape[1]!=Xte.shape[1]: errors.append("Train/test feature count mismatch")
    if Xtr.shape[0]!=ytr.shape[0] or Xte.shape[0]!=yte.shape[0]: errors.append("Feature/target row mismatch")
    report={"validation_status":"PASSED" if not errors else "FAILED","X_train_shape":list(Xtr.shape),"X_test_shape":list(Xte.shape),"missing_X_train":int(np.isnan(Xtr).sum()),"missing_X_test":int(np.isnan(Xte).sum()),"errors":errors}
    os.makedirs("artifacts",exist_ok=True); json.dump(report,open("artifacts/preprocessing_summary_report.json","w"),indent=4)
    print("[SUCCESS] Output Validation PASSED." if not errors else "[ERROR] Output Validation FAILED.")
    return not errors
if __name__ == "__main__":
    if not validate_preprocessing_outputs(): raise SystemExit(1)
