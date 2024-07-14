import math

import pandas as pd
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot

df = pd.read_csv("homeprices_2.csv")
df.bedrooms = df.bedrooms.fillna(math.floor(df.bedrooms.median()))

reg = linear_model.LinearRegression()
print([df.area, df.bedrooms,df.age])
reg.fit(df[['area', 'bedrooms','age']],df.price)
print(f"Weights: {reg.coef_}")
print(f"Bias: {reg.intercept_}")

print(f"Price value for area:{3000}, bedrms:{3}, age:{40} is {reg.predict([[3000,3,40]])}")