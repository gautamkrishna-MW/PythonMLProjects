
from sklearn import linear_model
import pandas as pd

df = pd.read_csv('homeprices_3.csv')

# Create dummy variables and merge them with the original dataset
# Also, remove the town column
df_merged = pd.concat([df,pd.get_dummies(df.town)], axis='columns')

# Dropping additional dummy variable column to avoid the "Dummy Variable Trap" problem
df_final = df_merged.drop(['town',"west windsor"], axis="columns")
print(df_final)


# Create linear regression model
regModel = linear_model.LinearRegression()
X = df_final.drop(["price"], axis="columns")
Y = df_final.price
regModel.fit(X,Y)

print(regModel.predict([[2000,1,1]]))