

```python
import numpy as np
import pandas as pd

# 入力データファイルは"./input/"ディレクトリにあります.
# 例えば, これを実行すると (実行をクリックするか, Shift+Enterキーを押して) 入力ディレクトリ以下のすべてのファイルが一覧表示されます.

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

```

    /kaggle/input/titanic/train.csv
    /kaggle/input/titanic/gender_submission.csv
    /kaggle/input/titanic/test.csv
    

### 訓練データとテストデータの読み込み


```python
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.isnull().sum()
```




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




```python
# 訓練データを特徴量と目的変数に分ける
X_train = train.drop(['Survived'], axis=1)
y_train = train['Survived']
```


```python
X_test = test.copy()
```


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    

### 特徴量の作成
- 変数PassengerId, Name, Ticket, Cabinの削除
- 変数Sex, Embarkedにlabel encodingを適用する


```python
X_train = X_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
X_test = X_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
X_train.head(), y_train.head()
```




    (   Pclass     Sex   Age  SibSp  Parch     Fare Embarked
     0       3    male  22.0      1      0   7.2500        S
     1       1  female  38.0      1      0  71.2833        C
     2       3  female  26.0      0      0   7.9250        S
     3       1  female  35.0      1      0  53.1000        S
     4       3    male  35.0      0      0   8.0500        S,
     0    0
     1    1
     2    1
     3    1
     4    0
     Name: Survived, dtype: int64)




```python
from sklearn.preprocessing import LabelEncoder

# 変数Sex, Embarkedにlabel encodingを適用する
for c in ['Sex', 'Embarked']:
    le = LabelEncoder()
    # ラベルを覚えさせる
    le = le.fit(X_train[c].fillna('NA'))
    # 変換
    X_train[c] = le.transform(X_train[c].fillna('NA'))
    X_test[c] = le.transform(X_test[c].fillna('NA'))

    # 表示確認
for c in ['Sex', 'Embarked']:
    print(X_train[c].head())
```

    0    1
    1    0
    2    0
    3    0
    4    1
    Name: Sex, dtype: int64
    0    3
    1    0
    2    3
    3    3
    4    3
    Name: Embarked, dtype: int64
    

## 学習


```python
from xgboost import XGBClassifier

# モデルの作成および学習データを与えての学習
model = XGBClassifier(n_estimators=20, random_state=42)
model.fit(X_train, y_train)

# テストデータの予測値を確率で出力する
pred = model.predict_proba(X_test)[:, 1]
# テストデータの予測値を二値に変換する
pred_label = np.where(pred > 0.5, 1, 0)

# 提出用ファイルの作成
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission_first.csv', index=False)
```


```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# チューニングするパラメータ
tuned_parameters = {
    "learning_rate":[0.05, 0.1, 0.2],
    "n_estimators":[1, 25, 50, 100],     # 作成するデータセットの数＝モデルの数
    "max_depth":[3, 5, 7, 10],
    "min_child_weight":[3, 5, 7, 10],
}

clf = GridSearchCV(
    XGBClassifier(random_state=42),
    tuned_parameters,
    scoring="accuracy",
    cv=5,
    n_jobs=1
)

clf = clf.fit(X_train, y_train) # モデル作成
best_clf = clf.best_estimator_  # 最も精度がよいモデルを取得

print("Best Parameter : ", clf.best_params_)

# テストデータの予測値を確率で出力する
pred_best = best_clf.predict_proba(X_test)[:, 1]

# テストデータの予測値を二値に変換する
pred_best_label = np.where(pred_best > 0.5, 1, 0)
```

    Best Parameter :  {'learning_rate': 0.1, 'max_depth': 7, 'min_child_weight': 3, 'n_estimators': 100}
    

### 交差検証


```python
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold

# 各foldのスコアを保存するリスト
score_accuracy = []
score_logloss = []

# クロスバリデーションの実施
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for tr_idx, va_idx in kf.split(X_train):
    # 訓練データを訓練データとバリデーションデータに分ける
    tr_x, va_x = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    tr_y, va_y = y_train.iloc[tr_idx], y_train.iloc[va_idx]
    
    # 学習の実施
    best_clf.fit(tr_x, tr_y)
    
    # バリデーションの予測値を確率で出力する
    va_pred = best_clf.predict_proba(va_x)[:, 1]
    
    # バリデーションデータのスコアを計算する
    logloss = log_loss(va_y, va_pred)
    accuracy = accuracy_score(va_y, va_pred > 0.5)
    
    # foldのスコアを保存する
    score_logloss.append(logloss)
    score_accuracy.append(accuracy)

# 各foldのスコアの平均を出力する
logloss = np.mean(score_logloss)
accuracy = np.mean(score_accuracy)
print(f'logloss: {logloss:.4f}, accuracy: {accuracy:.4f}')
```

    logloss: 0.4235, accuracy: 0.8362
    


```python
# 提出用ファイルの作成
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_best_label})
submission.to_csv('submission.csv', index=False)
```
