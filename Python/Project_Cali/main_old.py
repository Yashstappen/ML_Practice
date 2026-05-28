import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


#1. Loading the dataset
housing = pd.read_csv("housing.csv")

#2. Creating Stratified test set
housing['income_cat'] = pd.cut(housing["median_income"],
                               bins = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels = [1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis = 1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis = 1)

# Working on the copy of our data
housing = strat_train_set.copy()

#3. Making the features and labels in our data
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis = 1)

#print(housing, housing_labels)

#4. Seperating numerical and categorical columns 
num_attr = housing.drop("ocean_proximity", axis=1).columns.tolist() #made a new dataset which only contains the numerical attributes and then got the columns and converted to lists
cat_attr = ["ocean_proximity"]

#5. Making the Pipeline 
#For our numerical attributes
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy = "median")),
    ("scaler", StandardScaler())
])

#For categorical attributes
cat_pipeline = Pipeline([ #Onehot because it creates a sparse matrix instead of just assigning numbers
    ("onehot", OneHotEncoder(handle_unknown = "ignore")) #If there's any unknown values during testing we ignore it
])

#Constructing the final pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attr),
    ("cat", cat_pipeline, cat_attr)
])

#Apply fit transform 
housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared)

# 7. Testing/Training different models and select one

#Linear regression model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_pred = lin_reg.predict(housing_prepared)
lin_rmses = -cross_val_score(lin_reg, housing_prepared, housing_labels, cv=10, scoring="neg_root_mean_squared_error")
# print(f"The rmse of Linear Regression is: {lin_rmse}")
print(pd.Series(lin_rmses).describe())#mean-69204.322755

#Decision Tree model
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_labels)
dec_pred = dec_reg.predict(housing_prepared)
dec_rmses = -cross_val_score(dec_reg, housing_prepared, housing_labels, cv=10, scoring="neg_root_mean_squared_error") #Usually Higher is better, but in RMSE lower is better, hence we use -ve, -49>-60 but 49<60
# print(f"The rmse of DecisionTreeRegressor is: {dec_rmses}")
print(pd.Series(dec_rmses).describe())#mean-69097.008945

#Random Forest Regressor model
RFR_reg = RandomForestRegressor()
RFR_reg.fit(housing_prepared, housing_labels)
RFR_pred = RFR_reg.predict(housing_prepared)
RFR_rmses = -cross_val_score(RFR_reg, housing_prepared, housing_labels, cv=10, scoring="neg_root_mean_squared_error")
# print(f"The rmse of RandomForestRegressor is: {RFR_rmse}")
print(pd.Series(RFR_rmses).describe())#mean-49467.057737

#Hence we can go forward with the Random forest regressor 