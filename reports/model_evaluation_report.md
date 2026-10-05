# Technical Evaluation Report: Student Score Predictor

## Executive Summary
This report evaluates an end-to-end Machine Learning Early Warning System designed to forecast secondary school student final academic performance (Portuguese language course) prior to semester evaluations. The model was trained without mid-term examination scores (G1 and G2) to eliminate data leakage and ensure real-world utility for early academic intervention.

Two models were developed and compared:
1. A first-principles Linear Regression model built from scratch using NumPy and Batch Gradient Descent.
2. An industry-standard Scikit-Learn Ordinary Least Squares Linear Regression model.

Both models achieved near-identical quantitative results on a held-out test set of 130 students, validating the mathematical correctness of the from-scratch implementation.

---

## Evaluation Metrics Summary

| Metric | From-Scratch Model (NumPy) | Scikit-Learn (OLS) | Difference |
| :--- | :--- | :--- | :--- |
| Mean Absolute Error (MAE) | 1.8438 | 1.8438 | 0.0000 |
| Root Mean Squared Error (RMSE) | 2.3831 | 2.3826 | 0.0005 |
| R-squared (R^2) | 0.1364 | 0.1368 | 0.0004 |
| Convergence Cost (MSE) | 3.5211 | 3.5208 | 0.0003 |

### Metric Interpretations
- **Mean Absolute Error (1.84 points)**: On a 0 to 20 grading scale, the model's predictions are within approximately 1.8 points of a student's actual final grade before the academic period begins.
- **Root Mean Squared Error (2.38 points)**: The higher RMSE relative to MAE reflects the presence of zero-grade outliers (students who dropped out or missed the examination).
- **R-squared (0.137)**: Demonstrates that demographic, family background, and study habit features explain approximately 13.7% of the variance in final grades without access to prior academic tests.

---

## Regularization Analysis: OLS vs. Ridge vs. Lasso

| Model Specification | Active Features | Zeroed Features | MAE | RMSE | R-squared (R^2) | Key Characteristic |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OLS Linear Regression** | 39 | 0 | 1.8438 | 2.3826 | 0.1368 (13.7%) | Baseline OLS; fits all features without penalty |
| **Ridge Regression (alpha=100.0)** | 39 | 0 | **1.7887** | 2.3249 | 0.1781 (17.8%) | L2 squared penalty; dampens weights evenly across all columns |
| **Lasso Regression (alpha=0.0631, CV)** | 31 | 8 | 1.8051 | 2.3277 | 0.1761 (17.6%) | L1 absolute penalty; eliminates 8 noisy features via 5-fold CV |
| **Lasso Regression (alpha=0.5, Sparse)** | **6** | **33** | 1.8005 | **2.2775** | **0.2112 (21.1%)** | Aggressive sparsity; isolates top 6 predictive drivers |

---

## Key Drivers of Academic Performance

### Top Positive Contributors
1. **Desire for Higher Education (`higher_yes`, +0.540)**: The single strongest positive predictor of academic achievement.
2. **Weekly Study Time (`studytime`, +0.373)**: Direct correlation between weekly preparation hours and higher marks.
3. **Urban Address (`address_U`, +0.270)**: Urban residency provides proximity to educational resources.
4. **Smaller Family Size (`famsize_LE3`, +0.258)**: Families with three or fewer members show higher average achievement per student.
5. **Educator Parent (`Fjob_teacher`, +0.230)**: Paternal employment in education positively influences performance.

### Top Negative Contributors
1. **Past Class Failures (`failures`, -0.881)**: The single largest negative predictor. Prior failures compound future academic difficulty.
2. **School Affiliation (`school_MS`, -0.550)**: Attending Mousinho da Silveira school correlates with lower marks compared to Gabriel Pereira.
3. **Gender Indicator (`sex_M`, -0.433)**: Male students show lower average marks in the Portuguese language course.
4. **Extra Educational Support (`schoolsup_yes`, -0.397)**: Reflects reverse causality; support is assigned specifically to students who are already struggling.
5. **Health Status (`health`, -0.296)**: Poor self-reported health is linked to decreased academic output.

---

## Pipeline and Artifacts

All models and preprocessing statistics are serialized in the `model/` and `dataset/processed/` directories:
- `model/scratch_model.joblib`: Weights (39,) and bias float.
- `model/student_model_sklearn.joblib`: Scikit-learn estimator and feature schema.
- `dataset/processed/means.csv` and `dataset/processed/stds.csv`: Standardization parameters.
