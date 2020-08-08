# 行と列数の確認

```python
import pandas as pd
# 行数と列数の確認
df.shape
```

### 書式

	df: データフレーム

### 例

```python
import pandas as pd

# トレーニングデータ、テストデータを読み込み
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

print(train.shape) # 学習用データ
print(test.shape) # 本番予測用データ
```

```python
(891, 12)
(418, 11)
```

### 説明
行列の順番にデータフレームの構造を表示する