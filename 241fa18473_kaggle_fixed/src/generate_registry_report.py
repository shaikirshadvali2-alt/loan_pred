import os,json
from mlflow.tracking import MlflowClient
from mlflow_config import configure_mlflow
def generate_registry_report():
    configure_mlflow(); client=MlflowClient(); name="Loan_Prediction_Production_Model"; versions=list(client.search_model_versions(f"name='{name}'")); champion=next((v for v in versions if v.current_stage=="Production"),None)
    if champion is None: print("[ERROR] No Production model found."); return
    run=client.get_run(champion.run_id); report={"registry_status":"READY_FOR_DEPLOYMENT","model_lineage":{"registered_name":name,"version":int(champion.version),"current_stage":champion.current_stage,"run_id":champion.run_id,"artifact_uri":champion.source},"performance_metrics":run.data.metrics,"hyperparameters":run.data.params,"preprocessing_dependency":"models/preprocessor.pkl","dataset_lineage":"data/raw/loan_data.csv"}
    os.makedirs("artifacts",exist_ok=True); json.dump(report,open("artifacts/production_model_report.json","w"),indent=4); print("[SUCCESS] Registry report generated.")
if __name__ == "__main__": generate_registry_report()
