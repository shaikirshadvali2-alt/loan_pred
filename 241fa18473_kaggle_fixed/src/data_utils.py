from pathlib import Path
import pandas as pd

TRAIN_NAME = 'train_u6lujuX_CVtuZ9i.csv'
TEST_NAME = 'test_Y3wMUE5_7gLdaTN.csv'
RAW_DIR = Path('data/raw')


def find_dataset_file(name):
    candidates = [RAW_DIR / name, Path(name)]
    candidates += list(Path('.').glob(f'archive*/{name}'))
    candidates += list(Path('.').glob(f'**/{name}'))
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def load_train():
    p = find_dataset_file(TRAIN_NAME)
    if p is None:
        raise FileNotFoundError(
            f"{TRAIN_NAME} not found. Put the Kaggle train CSV in data/raw/ or any archive folder inside this project."
        )
    print(f'[INFO] Using training dataset: {p}')
    return pd.read_csv(p), p


def load_test(required=False):
    p = find_dataset_file(TEST_NAME)
    if p is None:
        if required:
            raise FileNotFoundError(f'{TEST_NAME} not found.')
        return None, None
    print(f'[INFO] Using test dataset: {p}')
    return pd.read_csv(p), p
