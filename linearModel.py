
import pandas as pd
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot

df = pd.read_csv("homeprices.csv", index_col=False)

# Fit linear model
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

# line equation -> y = slope*x + intercept
slopeVal = reg.coef_
interceptVal = reg.intercept_
newAreas = range(2000,6000, 1000)
newDF = pd.DataFrame({'area':newAreas})
outPrices = reg.predict(newDF)
print(outPrices)

# Plotting
pyplot.scatter(df.area, df.price, color='red', marker="*")
pyplot.plot(newAreas, outPrices, color='blue')
pyplot.show(block=False)
pyplot.pause(5)
pyplot.close()
