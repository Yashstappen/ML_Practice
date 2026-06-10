import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


def build_pipeline(num_attr):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return pipeline


# 1. Loading the dataset
standings = pd.read_csv("constructor_standings.csv")
races = pd.read_csv("races.csv")
df = pd.merge(standings, races, on="raceId")

# 2. Getting the mid season and end season rounds
season_end = df.groupby("year")["round"].max()
mid_season = df.groupby("year")["round"].max() // 2

mid_season_rounds = df[df["round"] == df["year"].map(mid_season)]
season_end_rounds = df[df["round"] == df["year"].map(season_end)]

# 3. Merging the important columns from both the mid season and end season rounds
id = pd.merge(
    mid_season_rounds,
    season_end_rounds,
    on=["constructorId", "year"]
)

data = id[["points_x", "position_x", "wins_x", "round_x", "position_y"]]

# 4. Making the features and labels in our data
features = data.drop("position_y", axis=1)
labels = data["position_y"]

# 5. Splitting the data into training and testing sets
split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

for train_index, test_index in split.split(features, labels):
    data.loc[test_index].to_csv("input.csv", index=False)
    features = features.loc[train_index]
    labels = labels.loc[train_index]

# 6. Transforming the data using pipelines
pipeline = build_pipeline(features.columns)
clean_data = pipeline.fit_transform(features)

# 7. Training the model
RFR_model = RandomForestRegressor(random_state=42)
RFR_model.fit(clean_data, labels)
RFR_RMSE = -cross_val_score(
    RFR_model,
    clean_data,
    labels,
    cv=10,
    scoring="neg_root_mean_squared_error"
)
print("Random Forest Regressor RMSE:", pd.Series(RFR_RMSE).describe())

linear_model = LinearRegression()
linear_model.fit(clean_data, labels)
linear_RMSE = -cross_val_score(
    linear_model,
    clean_data,
    labels,
    cv=10,
    scoring="neg_root_mean_squared_error"
)
print("Linear Regression RMSE:", pd.Series(linear_RMSE).describe())

decision_model = DecisionTreeRegressor()
decision_model.fit(clean_data, labels)
decision_RMSE = -cross_val_score(
    decision_model,
    clean_data,
    labels,
    cv=10,
    scoring="neg_root_mean_squared_error"
)
print("Decision Tree RMSE:", pd.Series(decision_RMSE).describe())

KN_model = KNeighborsRegressor()
KN_model.fit(clean_data, labels)
KN_RMSE = -cross_val_score(
    KN_model,
    clean_data,
    labels,
    cv=10,
    scoring="neg_root_mean_squared_error"
)
print("K-Nearest Neighbors RMSE:", pd.Series(KN_RMSE).describe())