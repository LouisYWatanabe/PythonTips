# カテゴリー変数を質的変数に指定した値で変換（map()）

```python
# 欠損値の再確認
data.isnull().sum()
```
```
PassengerId       0
Survived        418
Pclass            0
Name              0
Sex               0
Age             263
SibSp             0
Parch             0
Ticket            0
Fare              1
Cabin          1014
Embarked          2
dtype: int64
```

```python
# 頻度（出現回数）をカウント
data['Embarked'].value_counts()
```

```
S    914
C    270
Q    123
Name: Embarked, dtype: int64
```

```python
# 欠損値はSとして補完
data['Embarked'].fillna(('S'), inplace=True)
# S=0 C=1 Q=2 にint型で変換
data['Embarked'] = data['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

# 変換後の型の確認
data['Embarked'].dtype
```

```
dtype('int64')
```

```python
# 結合していたデータを再度訓練データとテストデータに分割
train = data[:len(train)]
test = data[len(train):]

# 目的変数と説明変数に分割
y_train = train['Survived']    # 目的変数
X_train = train.drop('Survived', axis=1)    # 訓練データの説明変数
X_test = test.drop('Survived', axis=1)    # テストデータの説明変数
```