import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import numpy as np
df=pd.read_csv('C:/Users/Gautham/OneDrive/Desktop/EBMAMRS/ml/Datasets/IRIS.csv')

x=df.drop('species',axis=1)
y=df['species']
X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)

names = ['Logistic Regression ', "GradientBoostingClasifier", "RandomForestClassifier", "Decision_Tree_Classifier","SVC", "MLPClassifier","MultinomialClassifier"]
regressors = [
LogisticRegression(random_state=45),
GradientBoostingClassifier(n_estimators=12),
RandomForestClassifier(random_state=2),
DecisionTreeClassifier(random_state=42),
svm.SVC(),
MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=2)
,MultinomialNB()]

scores = []
mean_score=[]
for name, clf in zip(names, regressors):
    clf.fit(X_train,y_train)
    score = accuracy_score(y_test,clf.predict(X_test))
    mse= 1-score
    scores.append(score)
    mean_score.append(mse)
    
scores_df = pd.DataFrame()
scores_df['name           '] = names
scores_df['accuracy'] = scores
scores_df['Mean_squared_error'] = mean_score
print(scores_df.sort_values('accuracy', ascending= False))



'''OUTPUT
             name             accuracy  Mean_squared_error
0       Logistic Regression   0.977778            0.022222
1  GradientBoostingClasifier  0.977778            0.022222
2     RandomForestClassifier  0.977778            0.022222
3   Decision_Tree_Classifier  0.977778            0.022222
4                        SVC  0.977778            0.022222
5              MLPClassifier  0.977778            0.022222
6      MultinomialClassifier  0.600000            0.400000
'''