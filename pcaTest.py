
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

dgData = load_digits()
dfData = pd.DataFrame(dgData.data)

# Show Stats
print(dfData.describe())

# Scale data before using
scalerObj = StandardScaler()
xScaled = scalerObj.fit_transform(dfData)
yInp = dgData.target

# Split training data
xTrain, xTest, yTrain, yTest = train_test_split(xScaled, yInp, test_size=0.4)

# Run logistic regression
modelObj = LogisticRegression()
modelObj.fit(xTrain, yTrain)
print(f"LogReg Score: {modelObj.score(xTest, yTest)}")

# PCA
# pcaObj = PCA(0.95)
pcaObj = PCA(n_components=20)
xTrain_PCA = pcaObj.fit_transform(xTrain)
xTest_PCA = pcaObj.transform(xTest)
print(pcaObj.explained_variance_ratio_) # Shows dimension-wise variance

# Applying LogReg after PCA
modelObj.fit(xTrain_PCA, yTrain)
print(f"LogReg Score after PCA: {modelObj.score(xTest_PCA, yTest)}")