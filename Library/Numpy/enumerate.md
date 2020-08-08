# 文字列を数値に変換(indexに変換)

```python
import numpy as np
import pandas as pd

df = pd.read_csv("../data/iris.data", header=None)

# データの4行目（品種）を目的変数として取得し .valueでarrayに変換
y = df.iloc[:, 4].values

# unique()でy要素の個数ごとにenumerate()でindex付きデータとして取得
# 目的変数の品種「Setosa」を0、「Versicolor」を1「Virginica」を2に変換
for idx, cl in enumerate(np.unique(y)):
    y[y == cl] = idx

# 説明変数の種類の確認
print("Class labels:", np.unique(y))
```

```python
Class labels: [0 1 2]
```

### 説明
