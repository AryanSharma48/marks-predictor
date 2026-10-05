import os
import pytest
import numpy as np
import pandas as pd
import joblib

from src.preprocessing import load_raw_data, preprocess_data
from src.predict import ScorePredictor


@pytest.fixture
def base_paths():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "raw_csv": os.path.join(root, "dataset", "raw", "student-por.csv"),
        "processed_dir": os.path.join(root, "dataset", "processed"),
        "model_dir": os.path.join(root, "model")
    }


@pytest.fixture
def sample_student():
    return {
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


# =====================================================================
# 1. Preprocessing & Data Pipeline Tests
# =====================================================================

def test_raw_dataset_loading(base_paths):
    """Verify raw dataset loads with expected row/col count."""
    df = load_raw_data(base_paths["raw_csv"])
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (649, 33)
    assert 'G3' in df.columns


def test_preprocessing_no_data_leakage(base_paths):
    """Verify G1, G2, G3 are dropped from features to prevent leakage."""
    raw_df = load_raw_data(base_paths["raw_csv"])
    X_train, X_test, y_train, y_test, means, stds, feature_names = preprocess_data(raw_df)

    # Check shapes
    assert X_train.shape[1] == 39
    assert X_test.shape[1] == 39
    assert len(y_train) == 519
    assert len(y_test) == 130

    # Ensure no leakage
    assert 'G1' not in feature_names
    assert 'G2' not in feature_names
    assert 'G3' not in feature_names


def test_standardization_statistics(base_paths):
    """Verify X_train is standardized with mean ~ 0 and std ~ 1."""
    raw_df = load_raw_data(base_paths["raw_csv"])
    X_train, _, _, _, _, _, _ = preprocess_data(raw_df)

    col_means = X_train.mean().values
    col_stds = X_train.std().values

    # Check that training mean is ~0 and std is ~1
    np.testing.assert_allclose(col_means, 0, atol=1e-7)
    np.testing.assert_allclose(col_stds, 1.0, atol=1e-7)


# =====================================================================
# 2. Model Artifact Tests
# =====================================================================

def test_saved_model_artifacts_exist(base_paths):
    """Verify both trained model bundles exist and load correctly."""
    sklearn_path = os.path.join(base_paths["model_dir"], "student_model_sklearn.joblib")
    scratch_path = os.path.join(base_paths["model_dir"], "scratch_model.joblib")

    assert os.path.exists(sklearn_path), "Scikit-learn model artifact missing."
    assert os.path.exists(scratch_path), "Scratch model artifact missing."

    sklearn_bundle = joblib.load(sklearn_path)
    assert "model" in sklearn_bundle
    assert "feature_names" in sklearn_bundle
    assert len(sklearn_bundle["feature_names"]) == 39

    scratch_bundle = joblib.load(scratch_path)
    assert "weights" in scratch_bundle
    assert "bias" in scratch_bundle
    assert len(scratch_bundle["weights"]) == 39


# =====================================================================
# 3. Inference Engine Tests
# =====================================================================

def test_inference_valid_output_range(sample_student):
    """Verify prediction produces a valid academic grade in [0.0, 20.0]."""
    predictor_sklearn = ScorePredictor(model_type="sklearn")
    score_sklearn = predictor_sklearn.predict(sample_student)
    assert isinstance(score_sklearn, float)
    assert 0.0 <= score_sklearn <= 20.0

    predictor_scratch = ScorePredictor(model_type="scratch")
    score_scratch = predictor_scratch.predict(sample_student)
    assert isinstance(score_scratch, float)
    assert 0.0 <= score_scratch <= 20.0


def test_inference_scratch_sklearn_consistency(sample_student):
    """Verify scratch model prediction matches scikit-learn prediction closely."""
    pred_sklearn = ScorePredictor(model_type="sklearn").predict(sample_student)
    pred_scratch = ScorePredictor(model_type="scratch").predict(sample_student)

    # Models should agree within 0.1 grade point
    assert abs(pred_sklearn - pred_scratch) <= 0.1


# =====================================================================
# 4. Edge Case & Robustness Tests
# =====================================================================

def test_inference_extreme_inputs(sample_student):
    """Verify clamping behaves properly under extreme conditions."""
    extreme_struggling_student = sample_student.copy()
    extreme_struggling_student.update({
        'failures': 10,
        'absences': 100,
        'studytime': 0,
        'higher': 'no',
        'Dalc': 5,
        'Walc': 5
    })

    predictor = ScorePredictor(model_type="sklearn")
    score = predictor.predict(extreme_struggling_student)
    assert score >= 0.0, "Score should never drop below 0.0"

    extreme_high_student = sample_student.copy()
    extreme_high_student.update({
        'failures': 0,
        'absences': 0,
        'studytime': 4,
        'higher': 'yes',
        'Medu': 4,
        'Fedu': 4
    })
    score_high = predictor.predict(extreme_high_student)
    assert score_high <= 20.0, "Score should never exceed 20.0"


def test_inference_partial_or_missing_categories(sample_student):
    """Verify column alignment (.reindex) handles missing category keys gracefully."""
    minimal_student = {
        'age': 15,
        'studytime': 2,
        'failures': 0,
        'absences': 4
    }

    predictor = ScorePredictor(model_type="sklearn")
    score = predictor.predict(minimal_student)
    assert 0.0 <= score <= 20.0
