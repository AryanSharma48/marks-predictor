import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


class ScratchLinearRegression:
    """
    Linear Regression implemented from first principles using pure NumPy
    and Batch Gradient Descent with MSE loss.
    """
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.cost_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.cost_history = []

        for _ in range(self.epochs):
            # Forward pass
            y_pred = (X @ self.weights) + self.bias
            error = y_pred - y

            # Gradients
            dw = (1 / m) * (X.T @ error)
            db = (1 / m) * np.sum(error)

            # Parameter updates
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Compute cost J(w, b) = 1/(2m) * sum(error^2)
            current_cost = (1 / (2 * m)) * np.sum(error ** 2)
            self.cost_history.append(current_cost)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X @ self.weights) + self.bias


def load_processed_data(processed_dir: str):
    """Loads standardized train and test matrices from disk."""
    X_train_df = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test_df = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv"))['G3'].values
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv"))['G3'].values

    feature_names = list(X_train_df.columns)
    return X_train_df, X_test_df, y_train, y_test, feature_names


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Computes MAE, RMSE, and R2 score."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n[{model_name}] Performance Evaluation:")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R^2  : {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_and_save_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "dataset", "processed")
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    print(f"[Training] Loading processed data from: {processed_dir}")
    X_train_df, X_test_df, y_train, y_test, feature_names = load_processed_data(processed_dir)

    X_train = X_train_df.values
    X_test = X_test_df.values

    # 1. Train Scratch Model
    print("\n[Training] Training Scratch Linear Regression (NumPy Gradient Descent)...")
    scratch_model = ScratchLinearRegression(learning_rate=0.01, epochs=1000)
    scratch_model.fit(X_train, y_train)
    scratch_preds = scratch_model.predict(X_test)
    evaluate_model(y_test, scratch_preds, "Scratch Model")

    scratch_save_path = os.path.join(model_dir, "scratch_model.joblib")
    joblib.dump({
        "weights": scratch_model.weights,
        "bias": scratch_model.bias,
        "cost_history": scratch_model.cost_history,
        "feature_names": feature_names
    }, scratch_save_path)
    print(f"[Training] Saved scratch model to: {scratch_save_path}")

    # 2. Train Scikit-Learn Model
    print("\n[Training] Training Scikit-Learn Linear Regression...")
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train_df, y_train)
    sklearn_preds = sklearn_model.predict(X_test_df)
    evaluate_model(y_test, sklearn_preds, "Scikit-Learn Model")

    sklearn_save_path = os.path.join(model_dir, "student_model_sklearn.joblib")
    joblib.dump({
        "model": sklearn_model,
        "feature_names": feature_names
    }, sklearn_save_path)
    print(f"[Training] Saved scikit-learn model bundle to: {sklearn_save_path}")


if __name__ == '__main__':
    train_and_save_models()
