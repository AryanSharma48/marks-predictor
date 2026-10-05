import os
import pandas as pd
import numpy as np


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Loads raw student performance dataset (semicolon-delimited)."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")
    return pd.read_csv(data_path, sep=';')


def preprocess_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Executes full data preprocessing pipeline:
    1. Isolates target y (G3)
    2. Drops G1, G2, G3 to prevent data leakage (Early Warning scope)
    3. One-hot encodes categorical features with drop_first=True
    4. Splits into train and test sets
    5. Standardizes features using training mean and std
    """
    # 1. Target separation & Early Warning feature selection
    y = df['G3'].copy()
    X = df.drop(columns=['G1', 'G2', 'G3']).copy()

    # 2. One-hot encoding
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_names = list(X_encoded.columns)

    # 3. Train-Test Split (reproducible with random_state)
    shuffled_X = X_encoded.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    shuffled_y = y.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    split_idx = int(len(shuffled_X) * (1 - test_size))

    X_train = shuffled_X.iloc[:split_idx].copy()
    X_test = shuffled_X.iloc[split_idx:].copy()
    y_train = shuffled_y.iloc[:split_idx].copy()
    y_test = shuffled_y.iloc[split_idx:].copy()

    # 4. Z-Score Standardization (derived strictly from training data)
    means = X_train.mean()
    stds = X_train.std()
    
    # Avoid zero division if any feature has zero variance
    stds = stds.replace(0, 1.0)

    X_train_scaled = (X_train - means) / stds
    X_test_scaled = (X_test - means) / stds

    return X_train_scaled, X_test_scaled, y_train, y_test, means, stds, feature_names


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    means: pd.Series,
    stds: pd.Series,
    output_dir: str
):
    """Exports processed datasets and scaling statistics to disk."""
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    means.to_csv(os.path.join(output_dir, "means.csv"), header=False)
    stds.to_csv(os.path.join(output_dir, "stds.csv"), header=False)
    print(f"[Preprocessing] Successfully saved processed data and statistics to: {output_dir}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "dataset", "raw", "student-por.csv")
    output_path = os.path.join(base_dir, "dataset", "processed")

    print(f"[Preprocessing] Loading dataset from: {raw_path}")
    raw_df = load_raw_data(raw_path)
    print(f"[Preprocessing] Raw dataset shape: {raw_df.shape}")

    X_train, X_test, y_train, y_test, means, stds, feature_names = preprocess_data(raw_df)
    print(f"[Preprocessing] Processed X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"[Preprocessing] Total features: {len(feature_names)}")

    save_processed_data(X_train, X_test, y_train, y_test, means, stds, output_path)


if __name__ == '__main__':
    main()
