# 特定の区切りでデータを分割 binning

```python
import pandas as pd

x = [1, 7, 5, 4, 6, 3]

# pandasのcut関数でbinningを行う

# binの数を指定する場合
binned = pd.cut(x, 3, labels=False)
print(binned)
# [0 2 1 1 2 0] - 変換された値は3つのbinのどれに入ったかを表す
```

### 例

```python
x = [1, 7, 5, 4, 6, 3]

# pandasのcut関数でbinningを行う

# binの数を指定する場合
binned = pd.cut(x, 3, labels=False)
print(binned)
# [0 2 1 1 2 0] - 変換された値は3つのbinのどれに入ったかを表す

# binの範囲を指定する場合（3.0以下、3.0より大きく5.0以下、5.0より大きい）
bin_edges = [-float('inf'), 3.0, 5.0, float('inf')]
binned = pd.cut(x, bin_edges, labels=False)
print(binned)
# [0 2 1 1 2 0] - 変換された値は3つのbinのどれに入ったかを表す
```

### 説明
データを特定の区切りで区切ってLabelingするにはpandasのcutを使用します。