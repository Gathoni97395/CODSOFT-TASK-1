Python 3.13.1 (tags/v3.13.1:0671451, Dec  3 2024, 19:06:28) [MSC v.1942 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import pandas as pd
import numpy as np
import matplotlib as ptb
#import machine learning  libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
dl=pd.read_csv(r"C:\Users\USER\Desktop\data science\Titanic-Dataset.csv")
print(dl.head())
   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0            1         0       3  ...   7.2500   NaN         S
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
3            4         1       1  ...  53.1000  C123         S
4            5         0       3  ...   8.0500   NaN         S

[5 rows x 12 columns]
#eda (exploratory data analysis)
 print(dl.describe())
 
SyntaxError: unexpected indent
print(dl.describe())
       PassengerId    Survived      Pclass  ...       SibSp       Parch        Fare
count   891.000000  891.000000  891.000000  ...  891.000000  891.000000  891.000000
mean    446.000000    0.383838    2.308642  ...    0.523008    0.381594   32.204208
std     257.353842    0.486592    0.836071  ...    1.102743    0.806057   49.693429
min       1.000000    0.000000    1.000000  ...    0.000000    0.000000    0.000000
25%     223.500000    0.000000    2.000000  ...    0.000000    0.000000    7.910400
50%     446.000000    0.000000    3.000000  ...    0.000000    0.000000   14.454200
75%     668.500000    1.000000    3.000000  ...    1.000000    0.000000   31.000000
max     891.000000    1.000000    3.000000  ...    8.000000    6.000000  512.329200

[8 rows x 7 columns]
#DATA CLEANING AND VALIDATION
#check and handle missing data
dl_data=dl.drop(columns="Cabin",axis=1)
missing_data=dl_data.isnull().sum()
print(missing_data)
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Embarked         2
dtype: int64
#column cabin dropped
#chck the percentages of missing data in age, and embarked columns
missing_data_percentage=(dl_data.isnull().sum() / len(dl_data)) * 100
print(missing_data_percentage)
PassengerId     0.000000
Survived        0.000000
Pclass          0.000000
Name            0.000000
Sex             0.000000
Age            19.865320
SibSp           0.000000
Parch           0.000000
Ticket          0.000000
Fare            0.000000
Embarked        0.224467
dtype: float64
# drop empty rows in column embarked
dl_data.dropna(subset=["Embarked"], axis=0,inplace=True)
print(missing_data)
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Embarked         2
dtype: int64
missing_data=dl_data.isnull().sum()
print(missing_data)
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Embarked         0
dtype: int64
print(dl_data.dtypes)
PassengerId      int64
Survived         int64
Pclass           int64
Name            object
Sex             object
Age            float64
SibSp            int64
Parch            int64
Ticket          object
Fare           float64
Embarked        object
dtype: object
#handling categorical values
dl_data['Embarked']=dl_data['Embarked'].map({'C':0,'S':1,'Q':2})
dl_data['Sex']=dl_data['Sex'].map({'male':0, 'female':1})
print(dl_data.dtypes)
PassengerId      int64
Survived         int64
Pclass           int64
Name            object
Sex              int64
Age            float64
SibSp            int64
Parch            int64
Ticket          object
Fare           float64
Embarked         int64
dtype: object
#splitting the data into tesing and training data
features=['Pclass', 'Sex', 'Age','SibSp', 'Parch', 'Fare', 'Embarked']
X=dl_data[features]
y=dl_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
print(X_train.shape)
(711, 7)
print(y_train.shape)
(711,)
#no mismatch in columns
#impute missing values for column age
X_train['Age']= X_train['Age'].fillna(X_train['Age'].mean())
X_test['Age']=X_test['Age'].fillna(X_test['Age'].mean())
#model training data using linear regression
model=LinearRegression()
model.fit(X_train, y_train)
LinearRegression()
#import seaborn
sns.histplot(X_train['Age'])
Traceback (most recent call last):
  File "<pyshell#46>", line 1, in <module>
    sns.histplot(X_train['Age'])
NameError: name 'sns' is not defined
import seaborn as sns
sns.
import seaborn as sns
sns.histplot(X_train['Age'])
<Axes: xlabel='Age', ylabel='Count'>
plt.show()
Traceback (most recent call last):
  File "<pyshell#50>", line 1, in <module>
    plt.show()
NameError: name 'plt' is not defined
ptb.show()
Traceback (most recent call last):
  File "<pyshell#51>", line 1, in <module>
    ptb.show()
  File "C:\Users\USER\AppData\Roaming\Python\Python313\site-packages\matplotlib\_api\__init__.py", line 218, in __getattr__
    raise AttributeError(
AttributeError: module 'matplotlib' has no attribute 'show'
y_pred=model.predict(X_test)
print(y_pred)
[ 0.67651406  0.05225079  0.5829919   0.59957244  0.23108505  0.11346419
  0.54887511  0.97158685  0.9847842   0.09981751  0.55207655  0.17361164
  0.09987467  0.97488902  0.70369446  0.12417885  0.06720643  0.32687966
  0.99778151  0.10175568  0.43162657  0.34294325  0.16284738  0.16834542
  0.11657628  0.06058449  0.11192147  0.45107525  0.31351579  0.04624738
  0.99907209  0.31741237  0.12697693  0.37207153  0.11192147  0.03551036
  0.67713408  0.09135266  0.12381638  0.10170458  0.9407503  -0.04778341
  0.1736898   0.70862159  0.57206222  0.75649543  0.59652648  0.39518002
  0.58940481  0.95982551  0.22036026  0.24707456  0.74975285  0.3244659
  0.63197524  0.20862738  0.91887203  0.09981751  0.17350036  0.15888982
  0.78236996  0.78262254  0.97094382  0.28782485  0.31787997  0.08528456
  0.59902597  0.83443756  0.99655568  0.12321658  0.14572981  0.17361164
  0.98809018  0.7350631   0.80470431  0.06800156  0.70365659  0.64700119
  0.81946145  0.40462016  0.11165299  0.98869238  1.04465313  0.97651794
  1.0265581   0.11181019  0.13246255  0.11192147  0.16242945  0.13018339
  0.23446293  0.26452561  0.27056905  0.11192147  0.59652648  0.15237784
  1.074147    0.09612698  0.10716787  0.46520106  0.64311168  0.17826289
  1.07865109  0.03861199  0.11992735  0.68576964  0.60393399  0.12321355
  0.5188727   0.06101985  0.97451727  0.13485726  0.96936858  0.47952403
  0.61003777  1.01468781  0.15342295  0.03041066  0.12321658  0.10810162
  0.09917399  0.13680805  0.07965283  0.17931251  0.08968337  0.30341133
  0.11134413  0.28475525  0.08974192  0.4903812   1.01563574  0.41527337
  0.67390959  0.4143434   0.59928293  0.59658363  0.22396863  0.20624709
  0.87306711  0.88880504  0.27487269  0.12321355  0.48474631  0.45483212
  0.07936416  0.26869396  0.04628062  0.09979947  0.91897126 -0.00325903
  0.66715869  0.15686687  0.64467332  0.91891145  0.14310374  0.28591358
  0.96884147  0.10207141  0.75049742  0.9995994   0.16853323  0.35520641
  0.28782485  0.54543166  0.91975803  0.14589815  0.18997451 -0.01681074
  0.76054077  0.22518871  0.06916526  0.82769056  0.51352027  0.07344892
  0.11657628  0.09737187  0.12321355  0.44209972]
#checking for accuracy and fitness of model
mse=mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
Mean Squared Error: 0.11579911037177212
r2=r2_score(y_test,y_pred)
print(f"R Squared:{r2}")
R Squared:0.5170489649836478
plt.scatter(y_test, y_pred)
Traceback (most recent call last):
  File "<pyshell#59>", line 1, in <module>
    plt.scatter(y_test, y_pred)
NameError: name 'plt' is not defined
plb.scatter(y_test, y_pred, alpha=0.5)
Traceback (most recent call last):
  File "<pyshell#60>", line 1, in <module>
    plb.scatter(y_test, y_pred, alpha=0.5)
NameError: name 'plb' is not defined. Did you mean: 'ptb'?
ptb.scatter(y_test, y_pred, alpha=0.5)
Traceback (most recent call last):
  File "<pyshell#61>", line 1, in <module>
    ptb.scatter(y_test, y_pred, alpha=0.5)
  File "C:\Users\USER\AppData\Roaming\Python\Python313\site-packages\matplotlib\_api\__init__.py", line 218, in __getattr__
    raise AttributeError(
AttributeError: module 'matplotlib' has no attribute 'scatter'
import matplotlib.pyplot as ptb
ptb.scatter(y_test, y_pred, alpha=0.5)
<matplotlib.collections.PathCollection object at 0x0000028A38A8BA10>
ptb.xlabel("Actual Survival")
Text(0.5, 0, 'Actual Survival')
ptb.ylabel("Predicted Survival")
Text(0, 0.5, 'Predicted Survival')
ptb.title("Actual vs. Predicted Survival")
Text(0.5, 1.0, 'Actual vs. Predicted Survival')
>>> ptb.xticks([0, 1])
([<matplotlib.axis.XTick object at 0x0000028A401174D0>, <matplotlib.axis.XTick object at 0x0000028A401A9950>], [Text(0, 0, '0'), Text(1, 0, '1')])
>>> plt.yticks([0, 1])
Traceback (most recent call last):
  File "<pyshell#68>", line 1, in <module>
    plt.yticks([0, 1])
NameError: name 'plt' is not defined
>>> ptb.yticks([0, 1])
([<matplotlib.axis.YTick object at 0x0000028A40117A10>, <matplotlib.axis.YTick object at 0x0000028A44312490>], [Text(0, 0, '0'), Text(0, 1, '1')])
>>> ptb.grid(True)
>>> ptb.show()
>>> from sklearn.model_selection import LogisticRegression
Traceback (most recent call last):
  File "<pyshell#72>", line 1, in <module>
    from sklearn.model_selection import LogisticRegression
ImportError: cannot import name 'LogisticRegression' from 'sklearn.model_selection' (C:\Users\USER\AppData\Roaming\Python\Python313\site-packages\sklearn\model_selection\__init__.py)
>>> from sklearn.linear_model import LogisticRegression
>>> model=
SyntaxError: invalid syntax
>>> model=Logisticregression()
Traceback (most recent call last):
  File "<pyshell#75>", line 1, in <module>
    model=Logisticregression()
NameError: name 'Logisticregression' is not defined. Did you mean: 'LogisticRegression'?
>>> model=LogisticRegression()
>>> model.fit(X_train, y_train)
LogisticRegression()
>>> #checking accuracy of logisitc regression
>>> y_pred=model.predict(X_test)
>>> from sklearn.metrics import accuracy_score
>>> accuracy = accuracy_score(y_test, y_pred)
>>> print(f"Accuracy Score of model: {accuracy }")
Accuracy Score of model: 0.848314606741573
>>> import nbformat as nb
