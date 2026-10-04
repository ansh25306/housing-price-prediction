# Housing Price Prediction: EDA, Preprocessing & Baseline Modelling

End-to-end regression pipeline on the California Housing dataset (predict `median_house_value`).

**Notebook:** `housing_price_prediction.ipynb`

## What it covers
- **Cleaning & preprocessing:** median imputation (`total_bedrooms`), one-hot encoding (`ocean_proximity`), percentile clipping of outliers, standard scaling for the linear model, ratio feature engineering
- **EDA:** univariate (histograms, count plots), outlier box plots, bivariate (scatter, box, geographic map), correlation heatmap
- **Models compared (5-fold CV):** Linear Regression, Decision Tree, Random Forest, Histogram Gradient Boosting, plus randomised-search tuning of the best ensemble
- **Metrics:** RMSE, MAE, R²; residual diagnostics and permutation importance
- **No data leakage:** stratified split happens first; all fitted steps live in scikit-learn pipelines fit on training data only; test set used once

## Dataset
California Housing (1990 US census, 20,640 block groups), as packaged in
[ageron/handson-ml2](https://github.com/ageron/handson-ml2/tree/master/datasets/housing).

- Direct download: https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv
- Save it as `data/housing.csv` (the notebook also finds it in the repo root).
- If no local file is found, the notebook downloads this same file automatically.

## Run it
**Colab / Kaggle:** upload the notebook and Run all (the data is downloaded automatically if `data/housing.csv` is absent).

**Locally:**
```bash
pip install -r requirements.txt
# optional: put housing.csv in data/ (see Dataset above)
jupyter notebook housing_price_prediction.ipynb
```
Runtime is a few minutes on a laptop (most of it is cross-validation and the tuning search).

## Repo layout
```
housing_price_prediction.ipynb
requirements.txt
README.md
data/housing.csv      # add this file (see Dataset above)
```
