# データ型の確認

```python
import numpy as np
import pandas as pd
# データ型の確認
y.dtype
```

### 書式

	y: 変数

### 例

```python
import numpy as np
import pandas as pd

df = pd.read_csv("../data/iris.data", header=None)

# unique()でy要素の個数ごとにenumerate()でindex付きデータとして取得
# 目的変数の品種「Setosa」を0、「Versicolor」を1「Virginica」を2に変換
for idx, cl in enumerate(np.unique(y)):
    y[y == cl] = idx

# 3行目（花びらの長さ）と4行目（花びらの幅）を説明変数として取得し .valueでarrayに変換
X = df.iloc[:, [2,3]].values

# 特徴量のデータ型の確認
print("目的変数の型", y.dtype)
print("説明変数の型", X.dtype)
```

```python
目的変数の型 object
説明変数の型 float64
```

### 説明

