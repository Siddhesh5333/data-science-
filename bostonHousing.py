import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df=pd.read_csv('C:/Users/Gautham/OneDrive/Desktop/EBMAMRS/ml/Datasets/housing.csv',header=None,delimiter=r"\s+",names=column_names)


for col in column_names:
    if df[col].count()!=506:
        df[col].fillna(df[col].median(),inplace=True)

# print(mean_squared_error(y_test,y_pred))

#Dropping CHAS because it has discrete values
df=df.drop('CHAS',axis=1)

# Box plot diagram
'''
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in df.items():
    sns.boxplot(y=k, data=df, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()
'''

#Outliers percentage in every column
'''
for k, v in df.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(df)[0]
    print("Column %s outliers = %.2f%%" % (k, perc))
'''
#Outliers Results
'''
Column CRIM outliers = 13.04%
Column ZN outliers = 13.44%
Column INDUS outliers = 0.00%
Column NOX outliers = 0.00%
Column RM outliers = 5.93%
Column AGE outliers = 0.00%
Column DIS outliers = 0.99%
Column RAD outliers = 0.00%
Column TAX outliers = 0.00%
Column PTRATIO outliers = 2.96%
Column B outliers = 15.22%
Column LSTAT outliers = 1.38%
Column MEDV outliers = 7.91%
'''


#Dropping Outliers greater than 10%
df=df.drop('CRIM',axis=1)
df=df.drop('ZN',axis=1)
df=df.drop('B',axis=1)

# MEDV max value is greater than 50.Based on that, values above 50.00 may not help to predict MEDV
# lets remove MEDV value above 50
df= df[~(df['MEDV'] >= 50.0)]

#TRAINING AND TESTING
x=df.drop('MEDV',axis=1)
y=df['MEDV']

pca=PCA(n_components=2)
pca.fit(x)
#Splitting Data
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
names = ['Linear Regression', "KNN", "Linear_SVM","Gradient_Boosting", "Decision_Tree", "Random_Forest"]
regressors = [
    LinearRegression(),
    KNeighborsRegressor(n_neighbors=3),
    SVR(),
    GradientBoostingRegressor(n_estimators=100),
    DecisionTreeRegressor(max_depth=5),
    RandomForestRegressor(max_depth=5, n_estimators=100)]

scores = []
mean_score=[]
for name, clf in zip(names, regressors):
    clf.fit(x_train,y_train)
    score = clf.score(x_test,y_test)
    mse= mean_squared_error(y_test,clf.predict(x_test))
    scores.append(score)
    mean_score.append(mse)
    
scores_df = pd.DataFrame()
scores_df['name           '] = names
scores_df['accuracy'] = scores
scores_df['Mean_squared_error'] = mean_score
print(scores_df.sort_values('accuracy', ascending= False))


'''
     name             accuracy  Mean_squared_error
3  Gradient_Boosting  0.850026           11.596362
5      Random_Forest  0.813331           14.433736
4      Decision_Tree  0.773798           17.490518
0  Linear Regression  0.741478           19.989630
1                KNN  0.527942           36.500805
2         Linear_SVM  0.259971           57.220990
'''