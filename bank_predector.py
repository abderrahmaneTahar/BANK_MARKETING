# %%
import pandas as pd
df = pd.read_csv("../banks.csv")

df["default"]=df["default"].replace(to_replace=['no', 'yes'], value=[0, 1])
df["housing"]=df["housing"].replace(to_replace=['no', 'yes'], value=[0, 1])
df["loan"]=df["loan"].replace(to_replace=['no', 'yes'], value=[0, 1])
df["y"]=df["y"].replace(to_replace=['no', 'yes'], value=[0, 1])

df["month"]=df["month"].replace(to_replace=['oct', 'may', 'apr', 'jun', 'feb', 'aug', 'jan', 'jul', 'nov','sep', 'mar', 'dec'], value=[10,5,4,6,2,8,1,7,11,9,3,12])

df=pd.get_dummies(data=df, columns=["job","marital","contact","poutcome"])

df["education"] = df["education"].replace(to_replace=["unknown"],value=["secondary"])
df["education"] = df["education"].replace(to_replace=["primary","secondary","tertiary"],value=[1,2,3])
df.education.value_counts()
df.head()

x = df.drop(['y'],axis=1)

y = df.y
x.head()

from sklearn.model_selection import train_test_split
from sklearn import metrics
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=109)
#Import Gaussian Naive Bayes model



#%%
import matplotlib.pyplot as plt
import seaborn as sns


#%%
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# %%
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression()
logistic_regression.fit(xt,y_train)
y_pred = logistic_regression.predict(xs)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# %%
x.describe()


# %%
from sklearn.preprocessing import StandardScaler
import numpy as np
sc = StandardScaler()
sc.fit(X_train)
xt = sc.transform(X_train)
xs = sc.transform(X_test)

xt= pd.DataFrame(xt,columns=X_train.columns)
xs = pd.DataFrame(xs,columns=X_test.columns)

np.round(xt.describe(),2)
# %%
plt.hist(x.age);

# %%
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(xt, y_train)

#Predict the response for test dataset
y_pred = clf.predict(xs)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# %%
