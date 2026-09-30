import os, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import load_train

EXPECTED = ['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']
NUMERIC = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']
CATEGORIES = {
    'Gender': {'Male','Female'}, 'Married': {'Yes','No'},
    'Education': {'Graduate','Not Graduate'}, 'Self_Employed': {'Yes','No'},
    'Property_Area': {'Urban','Semiurban','Rural'}, 'Loan_Status': {'Y','N'}
}

def validate_schema(df):
    print(f'[INFO] Validating Kaggle loan dataset schema (Records: {len(df)})...')
    errors=[]
    missing=[c for c in EXPECTED if c not in df.columns]
    extra=[c for c in df.columns if c not in EXPECTED]
    if missing: errors.append('Missing columns: '+', '.join(missing))
    if extra: errors.append('Unexpected columns: '+', '.join(extra))
    for c in NUMERIC:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            errors.append(f'Datatype mismatch: {c} must be numeric')
    for c, allowed in CATEGORIES.items():
        if c in df.columns:
            invalid=set(df[c].dropna().astype(str).unique())-allowed
            if invalid: errors.append(f'Invalid values in {c}: {sorted(invalid)}')
    if 'Loan_ID' in df.columns and df['Loan_ID'].duplicated().any(): errors.append('Duplicate Loan_ID values found')
    if 'ApplicantIncome' in df.columns and (df['ApplicantIncome'].dropna()<0).any(): errors.append('Negative ApplicantIncome found')
    if 'LoanAmount' in df.columns and (df['LoanAmount'].dropna()<0).any(): errors.append('Negative LoanAmount found')
    nulls=df.isnull().sum(); nulls=nulls[nulls>0]
    if len(nulls):
        print('[INFO] Missing values detected; preprocessing imputers will handle them:')
        print(nulls.to_string())
    os.makedirs('artifacts', exist_ok=True)
    if errors:
        pd.DataFrame({'validation_error':errors}).to_csv('artifacts/schema_validation_errors.csv',index=False)
        print('[ERROR] Schema Validation FAILED')
        for e in errors: print(' -',e)
        return False
    print('[SUCCESS] Schema Validation PASSED.')
    return True

if __name__=='__main__':
    df,_=load_train()
    if not validate_schema(df): raise SystemExit(1)
