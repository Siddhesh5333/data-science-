import pandas as pd
df=pd.read_csv("D:\dataset\IRIS.csv")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

lreg=LogisticRegression()
rf=RandomForestClassifier(random_state=0)
gb=GradientBoostingClassifier(n_estimators=10)
dc=DecisionTreeClassifier(random_state=0)
sv=svm.SVC()
mlp=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(4,2),random_state=0)
gnb=GaussianNB
mnb=MultinomialNB


x=df.drop("species",axis=1)
y=df["species"]
# print(x)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0, train_size=0.3)
lreg.fit(x_train,y_train)
y_pred=lreg.predict(x_test)
print("LOGISTIC REGRESSION",accuracy_score(y_test, y_pred))

rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
print("RANDOM FOREST",accuracy_score(y_test, y_pred))

gb.fit(x_train,y_train)
y_pred=gb.predict(x_test)
print("GRADIENT BOOSTING",accuracy_score(y_test, y_pred))

dc.fit(x_train,y_train)
y_pred=dc.predict(x_test)
print("DECISION TREE",accuracy_score(y_test, y_pred))

sv.fit(x_train,y_train)
y_pred=sv.predict(x_test)
print("SVM",accuracy_score(y_test, y_pred))

mlp.fit(x_train,y_train)
y_pred=mlp.predict(x_test)
print("MLP",accuracy_score(y_test, y_pred))

gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)
print("GNB",accuracy_score(y_test, y_pred))

mnb.fit(x_train,y_train)
y_pred=mnb.predict(x_test)
print("MNB",accuracy_score(y_test, y_pred))

# Output
# LOGISTIC REGRESSION 0.9047619047619048
# RANDOM FOREST 0.9619047619047619
# GRADIENT BOOSTING 0.9523809523809523
# DECISION TREE 0.9428571428571428
# SVM 0.8857142857142857
# MLP 0.8476190476190476