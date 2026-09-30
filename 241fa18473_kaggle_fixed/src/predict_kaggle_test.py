import sys, joblib, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import load_test

def main():
    df, source = load_test(required=True)
    ids = df['Loan_ID'].copy()
    X = df.drop(columns=['Loan_ID'])
    pre = joblib.load('models/preprocessor.pkl')
    model_path = Path('models/random_forest_model.pkl')
    if not model_path.exists(): model_path = Path('models/random_forest_baseline.pkl')
    model = joblib.load(model_path)
    pred = model.predict(pre.transform(X))
    out = pd.DataFrame({'Loan_ID': ids, 'Loan_Status': pd.Series(pred).map({1:'Y',0:'N'})})
    Path('outputs').mkdir(exist_ok=True)
    out.to_csv('outputs/kaggle_test_predictions.csv', index=False)
    print(f'[SUCCESS] Predictions generated for {len(out)} rows from {source}')
    print('[OUTPUT] outputs/kaggle_test_predictions.csv')
if __name__=='__main__': main()
