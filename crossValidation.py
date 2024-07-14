import pandas as pd
import seaborn as sb
from matplotlib import pyplot
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def runModel(modelObj, xTrain, yTrain, xTest, ytest) -> float:
    modelObj.fit(xTrain, yTrain)
    return modelObj.score(xTest, ytest)

# Create pandas dataframe
digits = load_digits()
digitDF = pd.DataFrame(digits.data)
digitDF['target'] = digits.target

# Split train-test data
xInp = digitDF.drop('target',axis='columns')
yInp = digitDF['target']
xTrain, xTest, yTrain, yTest = train_test_split(xInp, yInp, train_size=0.8)

# Train SVM
modelSVM = SVC()
modelSVM.fit(xTrain, yTrain)
scoreSVM = modelSVM.score(xTest, yTest)

# Train Logistic Regressor
modelLog = LogisticRegression(solver="newton-cg")
modelLog.fit(xTrain, yTrain)
scoreLog = modelLog.score(xTest, yTest)

# Train Random Forest
modelRFC = RandomForestClassifier()
modelRFC.fit(xTrain, yTrain)
scoreRFC = modelRFC.score(xTest, yTest)

print(f"Test Scores SVM:{scoreSVM}\t LogReg:{scoreLog}\t RFC:{scoreRFC}")

# Cross-validation using stratified k-fold to avoid bias due to random selection
from sklearn.model_selection import StratifiedKFold
cvObj = StratifiedKFold(n_splits=10)

scoreArrSVM=[]
scoreArrLog=[]
scoreArrRFC=[]
for trainIdx, testIdx in cvObj.split(xInp, yInp):
    xTrain, xTest = xInp.loc[trainIdx.tolist(),:], xInp.loc[testIdx.tolist(),:]
    yTrain, yTest = yInp.loc[trainIdx.tolist()], yInp.loc[testIdx.tolist()]
    scoreLog = runModel(modelLog, xTrain, yTrain, xTest, yTest)
    scoreSVM = runModel(modelSVM, xTrain, yTrain, xTest, yTest)
    scoreRFC = runModel(modelRFC, xTrain, yTrain, xTest, yTest)
    print(f"Test Scores SVM:{scoreSVM}\t LogReg:{scoreLog}\t RFC:{scoreRFC}")

    scoreArrSVM.append(scoreSVM)
    scoreArrLog.append(scoreLog)
    scoreArrRFC.append(scoreRFC)

from statistics import mean
meanScoreSVM = mean(scoreArrSVM)
meanScoreLog = mean(scoreArrLog)
meanScoreRFC = mean(scoreArrRFC)
print(f"Average Test Scores SVM:{meanScoreSVM}\t LogReg:{meanScoreLog}\t RFC:{meanScoreRFC}")

# SKLearn k-fold CV
from sklearn.model_selection import cross_val_score
crossValLog = mean(cross_val_score(modelLog, xInp, yInp))
crossValSVM = mean(cross_val_score(modelSVM, xInp, yInp))
crossValRFV = mean(cross_val_score(modelRFC, xInp, yInp))
print(f"Average CV Test Scores SVM:{crossValSVM}\t LogReg:{crossValLog}\t RFC:{crossValRFV}")