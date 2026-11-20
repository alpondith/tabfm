# Google TabFM vs. Traditional ML Models: California Housing Benchmark
### Rigorous Multi-Paradigm Tabular Valuation Benchmark

---

### Objectives
- **Tabular Foundation Model Evaluation**: Benchmark Google Research's zero-shot TabFM (`TabFMRegressor`) against traditional learning paradigms (Linear Regression, Random Forest, and LightGBM) on California Housing valuation without hyperparameter tuning.
- **Standardized Diagnostic Reproducibility**: Provide structured, standalone notebooks for exploratory data analysis, individual model diagnostics, and unified multi-model comparisons with zero execution warnings and bitwise reproducibility.

---

## How to Run

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

When opening any notebook in Antigravity or VS Code, select the **`.venv` (Python 3.12)** kernel.

---

## File Guide

- **[`notebooks/main.ipynb`](notebooks/main.ipynb)**: Unified 4-way benchmark directly comparing Linear Regression, Random Forest, LightGBM, and TabFM under identical context and query conditions.
- **[`notebooks/eda.ipynb`](notebooks/eda.ipynb)**: Comprehensive dataset deep-dive covering descriptive statistics, programmatic data dictionary, correlation heatmaps, and geospatial coordinate mapping.
- **[`notebooks/linear_regression.ipynb`](notebooks/linear_regression.ipynb)**: Standalone Ordinary Least Squares (OLS) baseline with standardized coefficient weights ($\beta$) and residual diagnostics.
- **[`notebooks/random_forest.ipynb`](notebooks/random_forest.ipynb)**: Standalone Random Forest ensemble (100 trees) with Gini feature importances (MDI) and subgroup error analysis.
- **[`notebooks/lightgbm.ipynb`](notebooks/lightgbm.ipynb)**: Standalone LightGBM Regressor (100 trees) evaluating split count and information gain rankings.
- **[`notebooks/tabfm.ipynb`](notebooks/tabfm.ipynb)**: Standalone Google Research TabFM evaluation using zero-shot in-context learning without gradient descent.
- **`data/dataset.csv`**: 1990 California Housing census dataset (20,640 records).

---

## License
Distributed under the MIT License.
