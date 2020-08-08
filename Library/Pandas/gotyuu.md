# 特定の単語の抽出・集計

```python
import pandas as pd
# .より前の敬称を抽出し、重複を省く
set(train.Name.str.extract(' ([A-Za-z]+)\.', expand=False))
# 敬称をcountする
train.Name.str.extract(' ([A-Za-z]+)\.', expand=False).value_counts()
```

### 書式

	df: データフレーム

### 例

```python
import pandas as pd

# トレーニングデータ、テストデータを読み込み
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

# trainとtestを縦に連結
df_full = pd.concat([train, test], axis=0, sort=False)

# .より前の敬称を抽出し、重複を省く
set(train.Name.str.extract(' ([A-Za-z]+)\.', expand=False))
# 敬称をcountする
train.Name.str.extract(' ([A-Za-z]+)\.', expand=False).value_counts()

```

```python
{'Capt',
 'Col',
 'Countess',
 'Don',
 'Dr',
 'Jonkheer',
 'Lady',
 'Major',
 'Master',
 'Miss',
 'Mlle',
 'Mme',
 'Mr',
 'Mrs',
 'Ms',
 'Rev',
 'Sir'}

Mr          517
Miss        182
Mrs         125
Master       40
Dr            7
Rev           6
Col           2
Mlle          2
Major         2
Capt          1
Mme           1
Don           1
Countess      1
Lady          1
Ms            1
Sir           1
Jonkheer      1
Name: Name, dtype: int64
```

### 説明
データフレームの要約統計量を表示する