import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def setup_style():
    """Sets a clean, modern aesthetic for all figures."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.dpi"] = 300


def plot_target_distribution(raw_df: pd.DataFrame, output_dir: str):
    """Plot 1: Target Grade Distribution with KDE & Zero-Outlier highlight."""
    plt.figure(figsize=(8, 5))
    ax = sns.histplot(raw_df['G3'], kde=True, discrete=True, color="#2b5c8f", edgecolor="black", alpha=0.7)

    # Highlight the zero outliers
    zero_count = (raw_df['G3'] == 0).sum()
    ax.annotate(
        f'Absence / Dropout Spike\n({zero_count} students at 0)',
        xy=(0, zero_count),
        xytext=(3, zero_count + 15),
        arrowprops=dict(facecolor='#d9534f', shrink=0.08, width=1.5, headwidth=7),
        fontsize=10,
        fontweight='bold',
        color='#d9534f'
    )

    plt.title("Distribution of Final Student Grades (G3)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Final Grade (0 - 20 Scale)", fontsize=12)
    plt.ylabel("Number of Students", fontsize=12)
    plt.xlim(-1, 21)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "1_target_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def plot_correlation_bar(raw_df: pd.DataFrame, output_dir: str):
    """Plot 2: Correlation of Numeric Features with G3."""
    plt.figure(figsize=(9, 5))
    corrs = raw_df.corr(numeric_only=True)['G3'].drop(labels=['G1', 'G2', 'G3']).sort_values()

    colors = ['#d9534f' if c < 0 else '#2b5c8f' for c in corrs.values]
    ax = sns.barplot(x=corrs.values, y=corrs.index, palette=colors)

    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title("Correlation of Pre-Semester Features with Final Grade (G3)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Pearson Correlation Coefficient (r)", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "2_feature_correlations.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def plot_cost_convergence(cost_history: list, output_dir: str):
    """Plot 3: Training Loss Curve (First-Principles Gradient Descent)."""
    plt.figure(figsize=(8, 5))
    epochs = range(len(cost_history))

    plt.plot(epochs, cost_history, color="#1f77b4", linewidth=2.2, label="MSE Cost J(w, b)")
    plt.scatter([0, 500, len(cost_history)-1],
                [cost_history[0], cost_history[500], cost_history[-1]],
                color="#d9534f", zorder=5)

    plt.annotate(f"Initial: {cost_history[0]:.2f}", xy=(0, cost_history[0]), xytext=(50, cost_history[0]-5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    plt.annotate(f"Converged: {cost_history[-1]:.2f}", xy=(len(cost_history)-1, cost_history[-1]), xytext=(700, cost_history[-1]+15),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.title("Gradient Descent Cost Convergence Curve (From Scratch)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Epochs (Iterations)", fontsize=12)
    plt.ylabel("Mean Squared Error Cost", fontsize=12)
    plt.ylim(0, max(cost_history) + 5)
    plt.legend(frameon=True)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "3_cost_convergence.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def plot_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray, output_dir: str):
    """Plot 4: Actual vs Predicted Scatter with Ideal Diagonal Line."""
    plt.figure(figsize=(7, 6))

    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color="#2b5c8f", s=60, edgecolor="white")

    # 45-degree ideal fit line
    min_val, max_val = 0, 20
    plt.plot([min_val, max_val], [min_val, max_val], color="#d9534f", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")

    plt.title("Actual vs. Predicted Final Grade on Test Set (N=130)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Actual Grade (y_test)", fontsize=12)
    plt.ylabel("Predicted Grade (y_pred)", fontsize=12)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend(loc="upper left")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "4_actual_vs_predicted.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def plot_residuals(y_test: np.ndarray, y_pred: np.ndarray, output_dir: str):
    """Plot 5: Residual Error Distribution."""
    residuals = y_pred - y_test
    plt.figure(figsize=(8, 5))

    sns.histplot(residuals, kde=True, color="#388e3c", edgecolor="black", alpha=0.6)
    plt.axvline(0, color="#d9534f", linestyle="--", linewidth=1.8, label=f"Mean Error: {np.mean(residuals):.2f}")

    plt.title("Residuals Distribution (Predicted - Actual)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Residual Error (Grade Points)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, "5_residuals_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def plot_feature_importance(model, feature_names: list, output_dir: str):
    """Plot 6: Standardized Regression Coefficients (Feature Drivers)."""
    plt.figure(figsize=(10, 6))

    coef_series = pd.Series(model.coef_, index=feature_names).sort_values()
    top_and_bottom = pd.concat([coef_series.head(5), coef_series.tail(5)])

    colors = ['#d9534f' if val < 0 else '#2b5c8f' for val in top_and_bottom.values]
    sns.barplot(x=top_and_bottom.values, y=top_and_bottom.index, palette=colors)

    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title("Top Predictors of Final Grade (Linear Regression Weights)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Standardized Weight (Impact on G3)", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "6_feature_importance.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def plot_regularization_comparison(base_dir: str, output_dir: str):
    """Plot 7: Comparison of OLS, Ridge, and Lasso Regularization on Early Detection."""
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

    processed_dir = os.path.join(base_dir, "dataset", "processed")
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv"))['G3'].values
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv"))['G3'].values

    models = {
        "OLS Linear": LinearRegression(),
        "Ridge (a=149)": Ridge(alpha=149.0),
        "Lasso (a=0.063)": Lasso(alpha=0.0631)
    }

    results = []
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        active_features = int(np.sum(mdl.coef_ != 0))

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2,
            "Active Features": active_features
        })

    res_df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # Metric 1: R2 Score
    sns.barplot(data=res_df, x="Model", y="R2 Score", ax=axes[0], color="#2b5c8f")
    axes[0].set_title("Variance Explained (R2 Score)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("R2 Score (Higher is Better)")
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    # Metric 2: RMSE
    sns.barplot(data=res_df, x="Model", y="RMSE", ax=axes[1], color="#d9534f")
    axes[1].set_title("Prediction Spread (RMSE)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("RMSE in Grade Points (Lower is Better)")
    axes[1].set_ylim(2.0, 2.45)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    # Metric 3: MAE
    sns.barplot(data=res_df, x="Model", y="MAE", ax=axes[2], color="#f0ad4e")
    axes[2].set_title("Average Error (MAE)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("MAE in Grade Points (Lower is Better)")
    axes[2].set_ylim(1.6, 1.95)
    for p in axes[2].patches:
        axes[2].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    # Metric 4: Active Features
    sns.barplot(data=res_df, x="Model", y="Active Features", ax=axes[3], color="#5cb85c")
    axes[3].set_title("Feature Sparsity (Active Features)", fontsize=11, fontweight="bold")
    axes[3].set_ylabel("Count of Non-Zero Features")
    for p in axes[3].patches:
        axes[3].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle("Early Detection Model Comparison: OLS vs. Ridge vs. Lasso", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "7_model_comparison_regularization.png")
    plt.savefig(save_path, bbox_inches='tight')
    early_save_path = os.path.join(output_dir, "early_warning_comparison.png")
    plt.savefig(early_save_path, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved: {save_path}")
    print(f"[Visualization] Saved: {early_save_path}")


def plot_endterm_comparison(base_dir: str, output_dir: str):
    """Comparison of OLS, Ridge, and Lasso Regularization on End-Term Prediction (with G1/G2)."""
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

    processed_dir = os.path.join(base_dir, "dataset", "processed", "endterm")
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv"))['G3'].values
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv"))['G3'].values

    models = {
        "OLS Linear": LinearRegression(),
        "Ridge (a=8)": Ridge(alpha=8.0),
        "Lasso (a=0.094)": Lasso(alpha=0.0943)
    }

    results = []
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        active_features = int(np.sum(mdl.coef_ != 0))

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2,
            "Active Features": active_features
        })

    res_df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # Metric 1: R2 Score
    sns.barplot(data=res_df, x="Model", y="R2 Score", ax=axes[0], color="#2b5c8f")
    axes[0].set_title("Variance Explained (R2 Score)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("R2 Score (Higher is Better)")
    axes[0].set_ylim(0.7, 0.92)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    # Metric 2: RMSE
    sns.barplot(data=res_df, x="Model", y="RMSE", ax=axes[1], color="#d9534f")
    axes[1].set_title("Prediction Spread (RMSE)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("RMSE in Grade Points (Lower is Better)")
    axes[1].set_ylim(0.7, 1.1)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    # Metric 3: MAE
    sns.barplot(data=res_df, x="Model", y="MAE", ax=axes[2], color="#f0ad4e")
    axes[2].set_title("Average Error (MAE)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("MAE in Grade Points (Lower is Better)")
    axes[2].set_ylim(0.5, 0.85)
    for p in axes[2].patches:
        axes[2].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    # Metric 4: Active Features
    sns.barplot(data=res_df, x="Model", y="Active Features", ax=axes[3], color="#5cb85c")
    axes[3].set_title("Feature Sparsity (Active Features)", fontsize=11, fontweight="bold")
    axes[3].set_ylabel("Count of Non-Zero Features")
    axes[3].set_ylim(0, 48)
    for p in axes[3].patches:
        axes[3].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle("End-Term Model Comparison (with G1 & G2): OLS vs. Ridge vs. Lasso", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "endterm_comparison.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def plot_overall_comparison(base_dir: str, output_dir: str):
    """Overall Comparison of Early Warning vs End-Term across all techniques."""
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

    # Early Warning Data
    p_early = os.path.join(base_dir, "dataset", "processed")
    X_tr_e = pd.read_csv(os.path.join(p_early, "X_train.csv"))
    X_te_e = pd.read_csv(os.path.join(p_early, "X_test.csv"))
    y_tr_e = pd.read_csv(os.path.join(p_early, "y_train.csv"))['G3'].values
    y_te_e = pd.read_csv(os.path.join(p_early, "y_test.csv"))['G3'].values

    # End-Term Data
    p_end = os.path.join(base_dir, "dataset", "processed", "endterm")
    X_tr_end = pd.read_csv(os.path.join(p_end, "X_train.csv"))
    X_te_end = pd.read_csv(os.path.join(p_end, "X_test.csv"))
    y_tr_end = pd.read_csv(os.path.join(p_end, "y_train.csv"))['G3'].values
    y_te_end = pd.read_csv(os.path.join(p_end, "y_test.csv"))['G3'].values

    early_models = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=149.0),
        "Lasso": Lasso(alpha=0.0631)
    }
    end_models = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=8.0),
        "Lasso": Lasso(alpha=0.0943)
    }

    records = []
    for model_name, mdl in early_models.items():
        mdl.fit(X_tr_e, y_tr_e)
        preds = mdl.predict(X_te_e)
        records.append({
            "Stage": "Early Warning (No G1/G2)",
            "Model": model_name,
            "R2 Score": r2_score(y_te_e, preds),
            "MAE": mean_absolute_error(y_te_e, preds),
            "Active Features": int(np.sum(mdl.coef_ != 0))
        })

    for model_name, mdl in end_models.items():
        mdl.fit(X_tr_end, y_tr_end)
        preds = mdl.predict(X_te_end)
        records.append({
            "Stage": "End-Term (With G1/G2)",
            "Model": model_name,
            "R2 Score": r2_score(y_te_end, preds),
            "MAE": mean_absolute_error(y_te_end, preds),
            "Active Features": int(np.sum(mdl.coef_ != 0))
        })

    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = {"Early Warning (No G1/G2)": "#2b5c8f", "End-Term (With G1/G2)": "#2ca02c"}

    # Metric 1: R2 Score
    sns.barplot(data=df, x="Model", y="R2 Score", hue="Stage", ax=axes[0], palette=palette)
    axes[0].set_title("Variance Explained (R2 Score)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("R2 Score (Higher is Better)")
    axes[0].set_ylim(0, 1.0)
    for p in axes[0].patches:
        if p.get_height() > 0:
            axes[0].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=9, xytext=(0, 2), textcoords='offset points')

    # Metric 2: MAE
    sns.barplot(data=df, x="Model", y="MAE", hue="Stage", ax=axes[1], palette=palette)
    axes[1].set_title("Mean Absolute Error (MAE)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("MAE in Grade Points (Lower is Better)")
    axes[1].set_ylim(0, 2.2)
    for p in axes[1].patches:
        if p.get_height() > 0:
            axes[1].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=9, xytext=(0, 2), textcoords='offset points')

    # Metric 3: Active Features
    sns.barplot(data=df, x="Model", y="Active Features", hue="Stage", ax=axes[2], palette=palette)
    axes[2].set_title("Feature Sparsity (Active Features)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Count of Non-Zero Features")
    axes[2].set_ylim(0, 48)
    for p in axes[2].patches:
        if p.get_height() > 0:
            axes[2].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=9, xytext=(0, 2), textcoords='offset points')

    for ax in axes:
        ax.set_xlabel("")
        ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=True, fontsize=11)
    plt.suptitle("Early Warning vs. End-Term: Regularization Comparison", fontsize=14, fontweight="bold", y=1.12)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "overall_comparison.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def main():
    setup_style()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "reports", "figures")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Raw Data
    raw_path = os.path.join(base_dir, "dataset", "raw", "student-por.csv")
    raw_df = pd.read_csv(raw_path, sep=';')

    # 2. Processed & Models
    model_bundle = joblib.load(os.path.join(base_dir, "model", "student_model_sklearn.joblib"))
    scratch_bundle = joblib.load(os.path.join(base_dir, "model", "scratch_model.joblib"))

    X_test_df = pd.read_csv(os.path.join(base_dir, "dataset", "processed", "X_test.csv"))
    y_test = pd.read_csv(os.path.join(base_dir, "dataset", "processed", "y_test.csv"))['G3'].values

    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]
    y_pred = model.predict(X_test_df)
    cost_history = scratch_bundle["cost_history"]

    # Generate all plots
    print("[Visualization] Generating complete visual suite...")
    plot_target_distribution(raw_df, output_dir)
    plot_correlation_bar(raw_df, output_dir)
    plot_cost_convergence(cost_history, output_dir)
    plot_actual_vs_predicted(y_test, y_pred, output_dir)
    plot_residuals(y_test, y_pred, output_dir)
    plot_feature_importance(model, feature_names, output_dir)
    plot_regularization_comparison(base_dir, output_dir)
    plot_endterm_comparison(base_dir, output_dir)
    plot_overall_comparison(base_dir, output_dir)
    print("\n[Visualization] All plots generated and saved successfully!")


if __name__ == '__main__':
    main()

