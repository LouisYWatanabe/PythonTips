# データフレームの連結

```python
import pandas as pd
# obj1とobj2を縦に連結
pd.concat([obj1, obj2], axis=0, sort=False)
# concat
## 基本 (縦に積む: カラムは各DataFrameの和集合
df = pd.concat([df_1, df_2, df_3])

## 横につなげる
df = pd.concat([df_1, df_2], axis=1)

## 各DataFrameに共通のカラムのみで積む
df = pd.concat([df_1, df_2, df_3], join='inner')
```

### 書式
	[]:連結するオブジェクトの指定
	 axis: 連結方向, `=0`縦、`=1`横
	 sort:False

### 例

```python
import pandas as pd

# トレーニングデータ、テストデータを読み込み
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
# trainとtestを縦に連結
df_full = pd.concat([train, test], axis=0, sort=False)
```

### 説明
データフレームの連結を行う
