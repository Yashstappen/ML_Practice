import os
import pandas as pd
import numpy as np
import joblib
 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attr):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return pipeline
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
    linear_model = LinearRegression()
    linear_model.fit(clean_data, labels)
    
    # 8. Saving the model and pipeline
    joblib.dump(linear_model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model trained and saved.")

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    
    data = pd.read_csv("input.csv")
    data = data.drop("position_y", axis=1)
    # data = pd.DataFrame([{"points_x": 25, "position_x": 1, "wins_x": 1, "round_x": 10}])
    clean_data = pipeline.transform(data)

    predictions = model.predict(clean_data)
    results = data.copy()
    results["predicted_position"] = predictions
    print(results.head())