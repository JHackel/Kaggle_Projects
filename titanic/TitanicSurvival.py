import pandas as pd

from titanic import KNNClassification as kNN

train = pd.read_csv("titanic_data/train.csv")
test = pd.read_csv("titanic_data/test.csv")

train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())
train['Sex'] = train['Sex'].map({'female':0,'male':1})
test['Sex'] = test['Sex'].map({'female':0,'male':1})

featuresTrain = train[['Pclass','Sex','Age','Fare']].values
targetTrain = train[['Survived']].values
featuresTest = test[['Pclass','Sex','Age','Fare']].values

predictions = kNN.kNNClassification(featuresTrain,targetTrain,featuresTest,5)