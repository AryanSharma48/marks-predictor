import os
import joblib
import numpy as np
import pandas as pd


class ScorePredictor:
    """
    Production inference engine for predicting student academic performance
    on Day 1 (Early Warning) using demographic and study habit features.
    """
    def __init__(self, model_type: str = "sklearn"):
        self.model_type = model_type
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.processed_dir = os.path.join(base_dir, "dataset", "processed")
        self.model_dir = os.path.join(base_dir, "model")

        self.means = None
        self.stds = None
        self.feature_names = None
        self.model = None
        self.scratch_weights = None
        self.scratch_bias = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Loads model bundles, means, and stds from disk."""
        # Load scaling stats
        means_path = os.path.join(self.processed_dir, "means.csv")
        stds_path = os.path.join(self.processed_dir, "stds.csv")

        # Read as Series with feature names as index
        self.means = pd.read_csv(means_path, header=None, index_col=0).squeeze()
        self.stds = pd.read_csv(stds_path, header=None, index_col=0).squeeze()

        if self.model_type == "sklearn":
            model_path = os.path.join(self.model_dir, "student_model_sklearn.joblib")
            bundle = joblib.load(model_path)
            self.model = bundle["model"]
            self.feature_names = bundle["feature_names"]
        elif self.model_type == "scratch":
            model_path = os.path.join(self.model_dir, "scratch_model.joblib")
            bundle = joblib.load(model_path)
            self.scratch_weights = bundle["weights"]
            self.scratch_bias = bundle["bias"]
            self.feature_names = bundle["feature_names"]
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Choose 'sklearn' or 'scratch'.")

    def preprocess_input(self, raw_data: dict) -> pd.DataFrame:
        """
        Transforms a raw student feature dictionary into the exact 39-feature
        standardized DataFrame expected by the model.
        """
        # 1. Convert to 1-row DataFrame
        df_raw = pd.DataFrame([raw_data])

        # Drop G1, G2, G3 if user accidentally supplied them
        df_raw = df_raw.drop(columns=['G1', 'G2', 'G3'], errors='ignore')

        # 2. One-Hot Encode categorical features
        df_encoded = pd.get_dummies(df_raw, drop_first=True)

        # 3. Column Alignment: guarantee all 39 training columns exist in exact order
        df_aligned = df_encoded.reindex(columns=self.feature_names, fill_value=0)

        # 4. Standardize using training statistics
        df_scaled = (df_aligned - self.means) / self.stds

        return df_scaled

    def predict(self, raw_data: dict) -> float:
        """
        Predicts final grade (0-20 scale) for a student profile.
        """
        df_scaled = self.preprocess_input(raw_data)

        if self.model_type == "sklearn":
            raw_prediction = self.model.predict(df_scaled)[0]
        else:
            raw_prediction = (df_scaled.values @ self.scratch_weights)[0] + self.scratch_bias

        # Clamp output to valid grade range [0, 20]
        final_grade = float(np.clip(raw_prediction, 0.0, 20.0))
        return round(final_grade, 2)


def main():
    # Demonstration test profile
    sample_student = {
        'school': 'GP',
        'sex': 'F',
        'age': 16,
        'address': 'U',
        'famsize': 'LE3',
        'Pstatus': 'T',
        'Medu': 4,
        'Fedu': 4,
        'Mjob': 'teacher',
        'Fjob': 'services',
        'reason': 'reputation',
        'guardian': 'mother',
        'traveltime': 1,
        'studytime': 3,
        'failures': 0,
        'schoolsup': 'no',
        'famsup': 'yes',
        'paid': 'no',
        'activities': 'yes',
        'nursery': 'yes',
        'higher': 'yes',
        'internet': 'yes',
        'romantic': 'no',
        'famrel': 4,
        'freetime': 3,
        'goout': 2,
        'Dalc': 1,
        'Walc': 1,
        'health': 5,
        'absences': 2
    }

    print("[Inference Demo] Initializing Predictor (Scikit-Learn)...")
    predictor_sklearn = ScorePredictor(model_type="sklearn")
    score_sklearn = predictor_sklearn.predict(sample_student)
    print(f"  -> Predicted Final Grade (Scikit-Learn): {score_sklearn} / 20.0")

    print("\n[Inference Demo] Initializing Predictor (Scratch Model)...")
    predictor_scratch = ScorePredictor(model_type="scratch")
    score_scratch = predictor_scratch.predict(sample_student)
    print(f"  -> Predicted Final Grade (Scratch Model): {score_scratch} / 20.0")


if __name__ == '__main__':
    main()
