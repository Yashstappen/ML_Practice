# Project Cali 🏡

A machine learning model that predicts California housing prices based on 
district-level features like location, income, and population.

## About
Built as my first end-to-end ML project using a real-world dataset. 
The model is trained once, saved to disk, and reused for inference — 
mimicking how ML models are deployed in production.

## How it works
1. **Training** — Data is cleaned, preprocessed through a full sklearn 
   pipeline, and used to train a Random Forest Regressor
2. **Inference** — New input data is passed through the saved pipeline 
   and model to generate price predictions
3. **Output** — Predictions are saved to `output.csv`

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (Pipeline, ColumnTransformer, RandomForestRegressor)
- Joblib (model persistence)

## Concepts Applied
- Stratified train/test splitting
- Handling missing data with SimpleImputer
- Encoding categorical features with OneHotEncoder
- Feature scaling with StandardScaler
- Model serialization and inference

## Dataset
California Housing Dataset — 20,000+ district records with features 
including median income, house age, ocean proximity, and more.