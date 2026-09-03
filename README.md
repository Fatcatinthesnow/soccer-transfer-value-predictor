# Football Transfer Value Predictor

Predicts players' current market value (€) from performance and profile stats using machine learning.

## Overview
This project builds an ML pipeline to estimate a player's transfer market value based on age, goals, assists, minutes played, appearances, and position. The pipeline covers data cleaning, exploratory analysis, feature engineering, and training/comparing four regression models. The best model is deployed in an interactive Gradio demo for live predictions.

## Data
[Football Players Transfer Fee Prediction Dataset](https://www.kaggle.com/datasets/khanghunhnguyntrng/football-players-transfer-fee-prediction-dataset) (Kaggle), pulled automatically at runtime via `kagglehub` — no manual download needed.

## Methodology
- **Target:** `current_value` (current market value in €)
- **Features:** age, goals, assists, minutes played, appearances, position (one-hot encoded)
- **Models compared:** K-Nearest Neighbors, Random Forest, Gradient Boosting, SVR
- **Evaluation:** train/test split (80/20), scored on MAE and R²

## Results

| Model                 | MAE (€)     | R²   |
|-----------------------|------------:|-----:|
| KNN (baseline)        | ~3,596,000  | 0.16 |
| Random Forest         | ~3,319,000  | 0.39 |
| Support Vector Machine| ~3,318,388  | -0.09 |
| Gradient Boosting     | ~3,114,000  | **0.42** |

**Gradient Boosting** performed best. Hyperparameter tuning via grid search did not meaningfully improve on default settings, suggesting the defaults were already well-suited to this data.

An R² of 0.42 means the model explains about 42% of the variation in player value. The remaining variance likely reflects factors not in the dataset — contract length, club prestige, league quality, and market sentiment — rather than model weakness alone.

## Demo
An interactive prediction tool is included, built with [Gradio](https://gradio.app), where you can input a player's stats and position to get a predicted market value.

## How to Run
Written in Python, designed to run in Google Colab.

1. Open the notebook in Colab (badge below)
2. Run all cells top to bottom — data downloads automatically via `kagglehub`
3. Required libraries: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `gradio`, `kagglehub` (install any missing ones with `!pip install`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fatcatinthesnow/soccer-transfer-value-predictor/blob/main/Transfermarket_Value_Prediction.ipynb)

## Limitations & Future Work
- Missing contextual features: contract length remaining, club/league prestige, transfer market sentiment
- Uses a current-value snapshot only, not the full historical valuation trajectory
- Hyperparameter tuning (grid search) did not improve on defaults within the range tested
- Future work: incorporate historical valuation trends, expand the hyperparameter search, try additional features (injury history, disciplinary record)
