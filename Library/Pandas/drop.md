# 列の削除

```python
import pandas as pd

# 不要な列の削除
train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
```

### 書式

	df: データフレーム
	- axis: 0だと行、 1なら列を削除する
	- inplace: Trueにすると、元のデータに変更が反映されます。

### 例

```python
import pandas as pd

# トレーニングデータ、テストデータを読み込み
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

# SexとEmbarkedのOne-Hotエンコーディング
train = pd.get_dummies(train, columns=['Sex', 'Embarked'])
test = pd.get_dummies(test, columns=['Sex', 'Embarked'])
# 不要な列の削除（行の場合はaxis=0）
train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
```

### 説明
列を削除する