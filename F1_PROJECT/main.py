import os
import pandas as pd
import numpy as np
import joblib
from urllib.request import urlopen
import json

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attr):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return pipeline

response = urlopen("https://api.openf1.org/v1/championship_teams?session_key=9947")
data = json.load(response)
df = pd.DataFrame(data)

if not os.path.exists(MODEL_FILE):

    # 1. Loading the dataset
    standings = pd.read_csv("constructor_standings.csv")
    races = pd.read_csv("races.csv")
    df = pd.merge(standings, races, on="raceId")

    # 2. Getting the mid season and end season rounds
    season_end = (df.groupby("year")["round"].max())
    mid_season = (df.groupby("year")["round"].max()//2)
    mid_season_rounds = df[df["round"]==df["year"].map(mid_season)]
    season_end_rounds = df[df["round"]==df["year"].map(season_end)]
    # print(mid_season_rounds.columns)
    # print(season_end_rounds.columns)

    # 3. Merging the important columns from both the mid season and end season rounds
    id = pd.merge(mid_season_rounds, season_end_rounds, on=["constructorId", "year"])
    data = id[["points_x", "position_x", "wins_x", "round_x", "position_y"]]

    # 4. Making the features and labels in our data
    features = data.drop("position_y", axis=1)
    labels = data["position_y"]

    # 5. Splitting the data into training and testing sets
    split = split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(features, labels):
        data.loc[test_index].to_csv("input.csv", index=False)
        features = features.loc[train_index]
        labels = labels.loc[train_index]

    # 6. Transforming the data using pipelines
    pipeline = build_pipeline(features.columns)
    clean_data = pipeline.fit_transform(features)

    # 7. training the model
    model = LinearRegression()
    model.fit(clean_data, labels)
    
    # 8. Saving the model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model trained and saved.")

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    
    prediction_df = pd.DataFrame({
        "points_x":df["points_current"],
        "position_x":df["position_current"],
        "wins_x": 0,
        "round_x": 12
    })

    clean_data = pipeline.transform(prediction_df)

    results = pd.DataFrame({
        "team" : df["team_name"],
        "predicted_position":model.predict(clean_data)
    })
    results["predicted_position"] = results["predicted_position"].round(2)
    predictions = model.predict(clean_data)
    # print(results.sort_values("predicted_position"))

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
    
    results["actual_position"] = results["team"].map(actual_2025)
    # print(results.sort_values("predicted_position"))

    # rmse = root_mean_squared_error(results["actual_position"], results["predicted_position"])
    # print("\nRMSE:", rmse)

x = np.arange(len(results["actual_position"]))
width = 0.4

plt.style.use("dark_background")
plt.figure(figsize=(10, 6))
plt.barh(x-width/2, results["actual_position"], height=width, label="Actual Position", color="red")
plt.barh(x+width/2, results["predicted_position"], height=width, label="Predicted Position", color="white")
plt.yticks(x, results["team"])
plt.xlabel("Position")
plt.ylabel("Teams")
plt.title("F1 Constructor Standings Prediction")
plt.legend()
plt.show()