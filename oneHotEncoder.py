
import pandas as pd
import sklearn.linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv('homeprices_3.csv')
df_labelEncoded = df

# Use label encoder to get labels
le = LabelEncoder()
encodedLabels = le.fit_transform(df.town)
df_labelEncoded['town'] = encodedLabels

# Use One hot encoder to generate dummy variables
oheObj = OneHotEncoder()
X = df_labelEncoded[['town','area']]
ctObj = ColumnTransformer([('town', OneHotEncoder(),[0])], remainder='passthrough')
X_dummy = ctObj.fit_transform(X)
print(X_dummy)
X = X_dummy[:,1:]
print(X)
Y = df_labelEncoded.price

linReg = sklearn.linear_model.LinearRegression()
linReg.fit(X,Y)
print(linReg.coef_)
print(linReg.predict([[1,0,2800]]))