import os, json, joblib, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import load_train

TARGET = 'Loan_Status'
DROP_COLS = ['Loan_ID']


def prepare_raw(df):
    df = df.drop_duplicates().copy()
    if TARGET not in df.columns:
        raise ValueError(f'Missing target column: {TARGET}')
    y = df[TARGET].map({'Y': 1, 'N': 0})
    if y.isna().any():
        raise ValueError('Loan_Status must contain only Y/N values.')
    X = df.drop(columns=[TARGET] + [c for c in DROP_COLS if c in df.columns])
    return X, y.astype(int)


def run_preprocessing():
    print('[INFO] Starting Kaggle Loan Prediction preprocessing...')
    df, source = load_train()
    X, y = prepare_raw(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('numeric', num_pipe, num_cols),
        ('categorical', cat_pipe, cat_cols)
    ])
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    np.save('data/processed/X_train_final.npy', X_train_final)
    np.save('data/processed/X_test_final.npy', X_test_final)
    np.save('data/processed/y_train.npy', y_train.to_numpy(dtype=np.int64))
    np.save('data/processed/y_test.npy', y_test.to_numpy(dtype=np.int64))
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    metadata = {
        'dataset_name': 'Kaggle Loan Prediction Problem Dataset',
        'source_file': str(source), 'target': TARGET,
        'target_mapping': {'N': 0, 'Y': 1},
        'dropped_identifier': 'Loan_ID',
        'train_shape': list(X_train_final.shape), 'test_shape': list(X_test_final.shape),
        'numerical_features': num_cols, 'categorical_features': cat_cols,
        'random_state': 42
    }
    with open('data/processed/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f'[INFO] Raw records: {len(df)} | Train: {len(y_train)} | Test: {len(y_test)}')
    print('[SUCCESS] Preprocessing completed successfully!')

if __name__ == '__main__':
    run_preprocessing()
