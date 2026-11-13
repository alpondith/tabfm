# Machine Learning: California Housing Prices

A structured workspace for machine learning experiments on the California Housing Prices dataset (1990 census data), featuring traditional machine learning pipelines and Google Research's zero-shot Tabular Foundation Model (**TabFM**).

---

## 💻 Standard Local Setup (venv)

This project uses standard Python virtual environment conventions.

### 1. Create & Activate Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Note: On CPU-only machines, you can optionally install the lightweight CPU wheel of PyTorch via `pip install torch --index-url https://download.pytorch.org/whl/cpu` before installing requirements).*

### 3. Select Kernel in IDE
When opening notebooks in Antigravity or VS Code:
- Click **Select Kernel** in the top-right corner.
- Choose **Python Environments** → **`.venv` (Python 3.12)**.

---

## 📁 Repository Structure

```text
ML/
├── .venv/                                # Local virtual environment (ignored in git)
├── data/
│   └── california_housing.csv        # California Housing dataset (20,640 records)
├── model/                                # Local TabFM model weights & configuration
├── notebooks/
│   ├── california_housing_rf.ipynb           # EDA, Scikit-Learn Preprocessing & Random Forest
│   ├── california_housing_lgbm.ipynb         # LightGBM Regressor (Default vs. Tuned GBDT)
│   ├── california_housing_tabfm.ipynb        # Google Research TabFM Foundation Model
│   ├── california_housing_rf_vs_tabfm.ipynb  # Head-to-Head Benchmark: Random Forest vs. TabFM
│   ├── california_housing_lgbm_vs_tabfm.ipynb # Head-to-Head Benchmark: LightGBM vs. TabFM
│   └── california_housing_benchmark_all_3.ipynb # 3-Way Benchmark: Random Forest vs. LightGBM vs. TabFM
├── .gitignore                                # Standard Python ignore rules
├── requirements.txt                          # Project dependencies
└── README.md                                 # Documentation
```

---

## 📓 Notebooks Overview

### 1. `notebooks/california_housing_rf.ipynb`
- **Exploratory Data Analysis (EDA)**: Inspection of distributions, missing values, and geographical mapping.
- **Feature Engineering**: Domain ratio calculations (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`).
- **Preprocessing Pipeline**: Scikit-learn `Pipeline` and `ColumnTransformer` with median imputation, standard scaling, and one-hot encoding.
- **Modeling**: Baseline Linear Regression vs. Random Forest Regressor.

### 2. `notebooks/california_housing_lgbm.ipynb`
- **Gradient Boosted Decision Trees (GBDT)**: Fast leaf-wise tree boosting with LightGBM.
- **Comparative Benchmarking**: Linear Regression vs. Random Forest vs. LightGBM (Default & Tuned).
- **Diagnostics & Feature Importance**: Actual vs. Predicted plots, residual densities, and Split Count vs. Information Gain feature importance.
- **Top Accuracy**: Outperforms Random Forest with ~$42,950 RMSE ($R^2 \approx 0.859$).

### 3. `notebooks/california_housing_tabfm.ipynb`
- **TabFM (Google Research)**: Zero-shot Tabular Foundation Model (6.5 GB weights).
- **In-Context Learning (ICL)**: Trained on synthetic structural causal graphs; predicts without gradient descent.
- **In-Depth Evaluation**: Standalone performance evaluation, residual diagnostics, and income subgroup error analysis.

### 4. `notebooks/california_housing_rf_vs_tabfm.ipynb`
- **Head-to-Head Benchmark**: Fair, apples-to-apples comparison between classical ML (Random Forest) and foundation model (TabFM).
- **Self-Contained**: Evaluates both models on the exact same test dataset and training context.
- **Comparative Diagnostics**: Side-by-side Actual vs. Predicted plots, residual distributions, and demographic subgroup error breakdowns.

### 5. `notebooks/california_housing_lgbm_vs_tabfm.ipynb`
- **Modern GBDT vs. Foundation Model**: Direct empirical comparison between LightGBM and Google Research's TabFM.
- **Controlled Benchmark**: Identical context samples ($N=500$) and evaluation queries ($N=300$).
- **Efficiency & Accuracy Analysis**: Compares latency, sample efficiency, throughput (50,000 vs. 0.6 samples/s), and demographic calibration across income tiers.

### 6. `notebooks/california_housing_benchmark_all_3.ipynb`
- **3-Way Definitive Benchmark**: Comprehensive comparison between Random Forest (Bagging), LightGBM (Boosting), and TabFM (Foundation Model).
- **Hybrid Blending**: Demonstrates meta-ensembling ($0.5 \times \text{LightGBM} + 0.5 \times \text{TabFM}$), achieving superior accuracy ($R^2 \approx 0.871$, MAE ~\$30.5k).
- **Diagnostics & Decision Matrix**: 4-way actual vs. predicted plots, residual error KDEs, income tier demographic breakdowns, and architectural selection guidelines.

---

## ⚡ Running on Google Colab

All notebooks can also run directly in **Google Colab**:
1. Open [Google Colab](https://colab.research.google.com).
2. Upload any notebook from `notebooks/`.
3. If running TabFM, select **Runtime → Change runtime type → T4 GPU** for hardware acceleration.
4. Run all cells (`Runtime` -> `Run all` or `Ctrl + F9`).

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
