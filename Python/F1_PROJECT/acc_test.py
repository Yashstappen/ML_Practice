import os
import pandas as pd
import numpy as np
import joblib
 
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

from urllib.request import urlopen
import json
import pandas as pd

response = urlopen(
    "https://api.openf1.org/v1/championship_teams?session_key=9947"
)

data = json.load(response)

df = pd.DataFrame(data)

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attr):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return pipeline
# Load training data
standings = pd.read_csv("constructor_standings.csv")
races = pd.read_csv("races.csv")
df_train = pd.merge(standings, races, on="raceId")

# Mid-season and final-season snapshots
season_end = df_train.groupby("year")["round"].max()
mid_season = season_end // 2

mid_season_rounds = df_train[df_train["round"] == df_train["year"].map(mid_season)]
season_end_rounds = df_train[df_train["round"] == df_train["year"].map(season_end)]

merged = pd.merge(
    mid_season_rounds,
    season_end_rounds,
    on=["constructorId", "year"]
)

data = merged[
    ["points_x", "position_x", "wins_x", "round_x", "position_y"]
]

# Features and labels
features = data.drop("position_y", axis=1)
labels = data["position_y"]

# Pipeline
pipeline = build_pipeline(features.columns)
clean_train = pipeline.fit_transform(features)

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(clean_train, labels)

# 2025 mid-season data
prediction_df = pd.DataFrame({
    "points_x": df["points_current"],
    "position_x": df["position_current"],
    "wins_x": 0,
    "round_x": 12
})

clean_test = pipeline.transform(prediction_df)

predictions = model.predict(clean_test)

results = pd.DataFrame({
    "team": df["team_name"],
    "midseason_position": df["position_current"],
    "predicted_final_position": predictions
})

print(results.sort_values("predicted_final_position"))

actual_2025 = {
    "McLaren": 1,
    "Mercedes": 2,
    "Red Bull Racing": 3,
    "Ferrari": 4,
    "Williams": 5,
    "Racing Bulls": 6,
    "Aston Martin": 7,
    "Haas F1 Team": 8,
    "Kick Sauber": 9,
    "Alpine": 10
}

results["actual"] = results["team"].map(actual_2025)
print(results)
# rmse = root_mean_squared_error(
#     results["actual"],
#     results["predicted_final_position"]
# )

# print("\nRMSE:", rmse)