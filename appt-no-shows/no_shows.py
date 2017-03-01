import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle

data = pd.read_csv('No-show-edited.csv')
y = pd.DataFrame(data,columns=['Status'])

data = data.drop(['Status'], axis=1)

data_train, data_test, y_train, y_test = train_test_split(data, y, test_size=.2)
#data_test, data_validation, y_test, y_validation = train_test_split(data_test_and_val, y_test_and_val, test_size=.5)

clf = SGDClassifier()
clf.fit(data_train, np.ravel(y_train))
predicted = clf.predict(data_test)
print(predicted)

# At time of comment - looks to be always predicting that the patient will show up

score = clf.score(data_test,y_test)
print(score)
