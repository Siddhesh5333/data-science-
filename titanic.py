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
df=pd.read_csv('C:/Users/Gautham/OneDrive/Desktop/EBMAMRS/ml/Datasets/Titanic.csv')

#Dropping unwanted Features
df=df.drop('PassengerId',axis=1)
df=df.drop('Cabin',axis=1)
df=df.drop('Name',axis=1)
df=df.drop('Ticket',axis=1)
# df=df.drop('Fare',axis=1)

sns.boxplot(df['Fare'])
# plt.show()

df['Fare'].fillna(df['Fare'].mean(),inplace=True)

#Categorical Values
le = LabelEncoder()
le.fit(df['Sex'])
df['Sex']=le.transform(df['Sex'])
le = LabelEncoder()
le.fit(df['Embarked'])
df['Embarked']=le.transform(df['Embarked'])
df['Age'].fillna(df['Age'].mean(),inplace=True)
# print(df.describe())
df1=df.drop('Survived',axis=1)
et = ExtraTreesClassifier()
et.fit(df1,df['Survived'])
# print(et.feature_importances_)

feat_imp=pd.Series(et.feature_importances_,index=df1.columns)
feat_imp.nlargest(7).plot(kind='barh')
# plt.show()
df=df.drop('Age',axis=1)
df=df.drop('Embarked',axis=1)
df=df.drop('Pclass',axis=1)

x=df.drop('Survived',axis=1)
y=df['Survived']

#Feature Selection
# bestfeatures = SelectKBest(score_func=chi2,k='all')
# bestfeatures.fit(x,y)
# dfscore=pd.DataFrame(bestfeatures.scores_)
# dfcolumns=pd.DataFrame(x.columns)
# featuresScores=pd.concat([dfcolumns,dfscore],axis=1)
# featuresScores.columns=['Features','Score']
# print(featuresScores)
from sklearn.ensemble import ExtraTreesRegressor
selector = ExtraTreesRegressor()
selector.fit(x, y)
# from imblearn.over_sampling import SMOTE
# sms=SMOTE(random_state=0)
# x,y=sms.fit_resample(x,y)
# # print(sms.fit_resample(X,y))
# from  imblearn.under_sampling import RandomUnderSampler
# rus=RandomUnderSampler(random_state=0)
# x,y=rus.fit_resample(x,y)
# print(rus.fit_resample(X,y))

df['SibSp']=pd.cut(df['SibSp'],2,labels=[0,1])
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size=0.3,random_state=0)
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
    clf.fit(X_Train,Y_Train)
    score = accuracy_score(Y_Test,clf.predict(X_Test))
    mse= mean_squared_error(Y_Test,clf.predict(X_Test))
    scores.append(score)
    mean_score.append(mse)
    
scores_df = pd.DataFrame()
scores_df['name           '] = names
scores_df['accuracy'] = scores
scores_df['Mean_squared_error'] = mean_score
print(scores_df.sort_values('accuracy', ascending= False))


#RESULTS:
'''
             name             accuracy  Mean_squared_error
0       Logistic Regression   1.000000            0.000000
1  GradientBoostingClasifier  1.000000            0.000000
2     RandomForestClassifier  1.000000            0.000000
3   Decision_Tree_Classifier  1.000000            0.000000
5              MLPClassifier  0.976190            0.023810
6      MultinomialClassifier  0.793651            0.206349
4                        SVC  0.547619            0.452381
'''