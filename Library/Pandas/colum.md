# 列の名前の確認

```python
import pandas as pd
# 列名の確認
df.columns
```

### 書式

	df: データフレーム

### 例

```python
import pandas as pd

# トレーニングデータ、テストデータを読み込み
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

print(train.columns) # トレーニングデータの列名
print('-' * 10) # 区切り線を表示
print(test.columns) # テストデータの列名
```

```python
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
----------
Index(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
       'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
```

### 説明
列の名前を表示する