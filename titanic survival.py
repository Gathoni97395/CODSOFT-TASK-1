Python 3.13.1 (tags/v3.13.1:0671451, Dec  3 2024, 19:06:28) [MSC v.1942 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import pandas as pd
import numpy as np
import matplotlib as ptb
#import machine learning libraries
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
dl=pd.read_csv(r"C:\Users\USER\Desktop\data science\Titanic-Dataset.csv")
print(dl.head())
   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0            1         0       3  ...   7.2500   NaN         S
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
3            4         1       1  ...  53.1000  C123         S
4            5         0       3  ...   8.0500   NaN         S

[5 rows x 12 columns]
printt(dl.describe())
Traceback (most recent call last):
  File "<pyshell#10>", line 1, in <module>
    printt(dl.describe())
NameError: name 'printt' is not defined. Did you mean: 'print'?
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
#checking and handling missing data
missing_data=dl.isnull().sum()
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
Cabin          687
Embarked         2
dtype: int64
#age, Cabin & Embarked columns have missing data
#drop cabin
dl_data=drop.(columns"Cabin", axis=1)
SyntaxError: invalid syntax
>>> dl_data=drop.columns("Cabin,axis=1)
...                      
SyntaxError: unterminated string literal (detected at line 1)
>>> dl_data=drop(columns="Cabin", axis=1)
...                      
Traceback (most recent call last):
  File "<pyshell#19>", line 1, in <module>
    dl_data=drop(columns="Cabin", axis=1)
NameError: name 'drop' is not defined
>>> dl_data=dl.drop(columns="Cabin",axis=1)
...                      
>>> #split the data into testing and training data
...                      
>>> features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
...                 X = dl_data[features]
...                      
SyntaxError: multiple statements found while compiling a single statement
>>> features=['Pclass', 'Sex', 'Age','SibSp', 'Parch', 'Fare', 'Embarked']
...                      
>>> X=dl_data[features]
...                      
>>> y=dl_data['survived']
...                      
Traceback (most recent call last):
  File "C:\Users\USER\AppData\Roaming\Python\Python313\site-packages\pandas\core\indexes\base.py", line 3805, in get_loc
    return self._engine.get_loc(casted_key)
  File "index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
  File "index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\\_libs\\hashtable_class_helper.pxi", line 7081, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\\_libs\\hashtable_class_helper.pxi", line 7089, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'survived'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<pyshell#25>", line 1, in <module>
    y=dl_data['survived']
  File "C:\Users\USER\AppData\Roaming\Python\Python313\site-packages\pandas\core\frame.py", line 4102, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Users\USER\AppData\Roaming\Python\Python313\site-packages\pandas\core\indexes\base.py", line 3812, in get_loc
    raise KeyError(key) from err
KeyError: 'survived'
y=dl_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.describe())
           Pclass         Age       SibSp       Parch        Fare
count  712.000000  572.000000  712.000000  712.000000  712.000000
mean     2.330056   29.498846    0.553371    0.379213   32.586276
std      0.824584   14.500059    1.176404    0.791669   51.969529
min      1.000000    0.420000    0.000000    0.000000    0.000000
25%      2.000000   21.000000    0.000000    0.000000    7.925000
50%      3.000000   28.000000    0.000000    0.000000   14.454200
75%      3.000000   38.000000    1.000000    0.000000   30.500000
max      3.000000   80.000000    8.000000    6.000000  512.329200
print(y_train.describe())
count    712.000000
mean       0.376404
std        0.484824
min        0.000000
25%        0.000000
50%        0.000000
75%        1.000000
max        1.000000
Name: Survived, dtype: float64
# impute missing values for columns, age and embarked
# for age
X_train['Age']= X_train['Age'].fillna(X_train['Age'].mean())
X_test['Age']=X_test['Age'].fillna(X_test['Age'].mean())
X_train.dropna(subset=['Embarked'], axis=0, inplace=True)
X_test.dropna(subset=["Embarked"], axis=0,inplace=True)
#data validation, to confirm missing values are handled
missing_data_X=X_train.isnull().sum()
missing _data_X_test=X_test.isnull().sum()
SyntaxError: invalid syntax
missing_data_X_test=X_test.isnull().sum()
print(missing_data_X)
Pclass      0
Sex         0
Age         0
SibSp       0
Parch       0
Fare        0
Embarked    0
dtype: int64
print(missing_data_X_test)
Pclass      0
Sex         0
Age         0
SibSp       0
Parch       0
Fare        0
Embarked    0
dtype: int64
#handling  categorical variables for columns sex and embarked
X_train['Sex'] = X_train['Sex'].map({'male': 0, 'female': 1})
X_test['Sex']=X_test['Sex'].map({'male':0, 'female':1})
X_train['Embarked']=X_train['Embarked'].map({'C':0,'S':1,'Q':2})
X_test['Embarked']=X_test['Embarked'].map({'C':0,'S':1,'Q':2})
#model training data using logistic regression
model=LinearRegression()
model.fit(X_train, y_train)
Traceback (most recent call last):
  File "<pyshell#51>", line 1, in <module>
    model.fit(X_train, y_train)
  File "C:\Users\USER\AppData\Roaming\Python\Python313\site-packages\sklearn\base.py", line 1389, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "C:\Users\USER\AppData\Roaming\Python\Python313\site-packages\sklearn\linear_model\_base.py", line 601, in fit
    X, y = validate_data(
  File "C:\Users\USER\AppData\Roaming\Python\Python313\site-packages\sklearn\utils\validation.py", line 2961, in validate_data
    X, y = check_X_y(X, y, **check_params)
  File "C:\Users\USER\AppData\Roaming\Python\Python313\site-packages\sklearn\utils\validation.py", line 1389, in check_X_y
    check_consistent_length(X, y)
  File "C:\Users\USER\AppData\Roaming\Python\Python313\site-packages\sklearn\utils\validation.py", line 475, in check_consistent_length
    raise ValueError(
ValueError: Found input variables with inconsistent numbers of samples: [710, 712]
# handle mismatch in values
print(X_train.shape)
(710, 7)
print(y_train.shape)
(712,)
