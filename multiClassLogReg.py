
from matplotlib import pyplot
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

digits = load_digits()
print(dir(digits))
print(digits.target_names)

# Display sample image
pyplot.imshow(digits.images[0],cmap='gray', vmin=0, vmax=np.max(digits.data[0]))
pyplot.show(block=False)
pyplot.pause(2)
pyplot.close()

# Split train-test data
xTrain, xTest, yTrain, yTest = train_test_split(digits.data, digits.target, train_size=0.6)
logReg = LogisticRegression()
logReg.fit(xTrain, yTrain)

# Warning: Long loop
skipLoop = True
if not skipLoop:
    for idx in range(len(xTest)):
        img = np.reshape(xTest[idx],(8,8))
        maxVal = np.max(xTest[idx])
        pyplot.imshow(img,cmap='gray', vmin=0, vmax=maxVal)
        pyplot.title(f"Number: {yTest[idx]} is predicted as {logReg.predict([xTest[idx]])}")
        pyplot.show(block=False)
        pyplot.pause(2)
        pyplot.close()

print(f"Final Score: {logReg.score(xTest,yTest)}")

# Confusion Matrix
from sklearn.metrics import confusion_matrix
yPred = logReg.predict(xTest)
cfMat = confusion_matrix(yTest,yPred)
print(f"Confusion Matrix: \n{cfMat}")

