from mlflow.tracking import MlflowClient
from mlflow_config import configure_mlflow
def automate_lifecycle():
    configure_mlflow(); client=MlflowClient(); name="Loan_Prediction_Production_Model"; metric="f1_score"
    versions=list(client.search_model_versions(f"name='{name}'"))
    if not versions: print("[ERROR] No registered model versions found."); return
    scored=[]
    for mv in versions:
        run=client.get_run(mv.run_id); scored.append((run.data.metrics.get(metric,0.0),mv))
        print(f"Version {mv.version}: {metric}={run.data.metrics.get(metric,0.0):.4f}")
    best_score,best=max(scored,key=lambda x:x[0])
    for score,mv in scored:
        stage="Production" if mv.version==best.version else "Archived"
        client.transition_model_version_stage(name=name,version=mv.version,stage=stage,archive_existing_versions=False)
    print(f"[SUCCESS] Version {best.version} promoted to Production ({metric}={best_score:.4f}).")
if __name__ == "__main__": automate_lifecycle()
