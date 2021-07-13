import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from google.colab import files
uploaded = files.upload()
input=list(uploaded.keys())
input=input[0]

dataset=pd.read_csv('Churn_Modelling.csv')
dataset.head()

dataset.info()

types = dataset.dtypes
print(types)

dataset.describe()

types = dataset.dtypes
print(types)

class_counts = dataset.groupby('NumOfProducts').size()

print(class_counts)

from matplotlib import pyplot
dataset.hist()
pyplot.show()

dataset.plot(kind='density' ,subplots=True, layout=(4,4), sharex=False)
pyplot.show()

# Extracting dependent and independent variables:
# Extracting independent variable:
X = dataset.iloc[:,3:13].values
# Extracting dependent variable:
y = dataset.iloc[:, 13].values


from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

print(X)

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

print(X)

from sklearn.preprocessing import OneHotEncoder
 from sklearn.compose import ColumnTransformer

 columnTransformer = ColumnTransformer([('Dummy', OneHotEncoder(), [1])],remainder='passthrough')
 X=columnTransformer.fit_transform(X)

print(X)
X

X = X[:, 1:]

print(X)
X

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_train)

print(y_train)

print(X_test)

print(y_test)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print(X_train)

print(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

svm = SVC()
svm.fit(X_train, y_train)
svm.score(X_test, y_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

from sklearn.model_selection import GridSearchCV
model_params = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    }
}

scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(X, y)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df

model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)
