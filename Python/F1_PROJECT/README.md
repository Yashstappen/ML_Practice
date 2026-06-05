# F1 Constructor Championship Predictor

A machine learning project built from scratch out of sheer curiosity and a love for F1. Given a constructor's mid-season standings, it predicts where they'll finish in the World Constructors' Championship (WCC) at the end of the season.

This is my second ML project, built independently without following a tutorial.

## What it does
- Takes mid-season data for any constructor as input
- Runs it through a trained Linear Regression model
- Outputs a predicted final WCC position

## Dataset
Ergast F1 Dataset (via Kaggle, by Vopani). Three CSV files:
- `constructor_standings.csv` — points, position, and wins per constructor per race
- `races.csv` — race metadata including year and round number

## Output
A predicted final championship position. A value close to 1.0 means the model expects the constructor to finish 1st.

## Tech stack
- Python, Pandas, NumPy
- scikit-learn (Linear Regression, Pipeline, StandardScaler, SimpleImputer)
- Joblib for model persistence
