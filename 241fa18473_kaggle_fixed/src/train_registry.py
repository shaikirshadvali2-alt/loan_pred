import os,numpy as np,mlflow,mlflow.sklearn
from mlflow_config import configure_mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
def train_and_register_model():
    configure_mlflow()
    Xtr=np.load("data/processed/X_train_final.npy"); Xte=np.load("data/processed/X_test_final.npy"); ytr=np.load("data/processed/y_train.npy"); yte=np.load("data/processed/y_test.npy")
    params={"n_estimators":200,"max_depth":15,"random_state":42}
    with mlflow.start_run(run_name="RandomForest_Registry"):
        mlflow.log_params(params); mlflow.log_param("model_family","RandomForest")
        m=RandomForestClassifier(**params).fit(Xtr,ytr); pred=m.predict(Xte); prob=m.predict_proba(Xte)[:,1]
        metrics={"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred,zero_division=0),"recall":recall_score(yte,pred,zero_division=0),"f1_score":f1_score(yte,pred,zero_division=0),"roc_auc":roc_auc_score(yte,prob)}; mlflow.log_metrics(metrics)
        if os.path.exists("models/preprocessor.pkl"): mlflow.log_artifact("models/preprocessor.pkl",artifact_path="preprocessing_pipeline")
        if os.path.exists("data/processed/dataset_metadata.json"): mlflow.log_artifact("data/processed/dataset_metadata.json",artifact_path="metadata")
        mlflow.sklearn.log_model(sk_model=m,name="random_forest_model",registered_model_name="Loan_Prediction_Production_Model")
        print(f"[SUCCESS] Model registered. F1={metrics['f1_score']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
if __name__ == "__main__": train_and_register_model()
