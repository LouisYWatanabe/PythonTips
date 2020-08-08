# 1次元配列の宣言


```python
import pandas as pd
# ndarrayクラス
array = np.array([1,2,3,4,5,6,7,8])
```

### 書式

	np.array([リスト])

### 引数

- <b>[リスト]</b>
数値、または文字列のリストを代入する

### 例

```python
import numpy as np

storages = [24, 3, 4, 23, 10, 12]
print(storages)

# ndarray配列を生成し、変数np_storagesに代入してください
np_storages = np.array(storages)
print(np_storages)

# 変数np_storagesの型を出力してください
print(type(np_storages))

```
```python
[24, 3, 4, 23, 10, 12]
[24  3  4 23 10 12]
<class 'numpy.ndarray'>
```

### 説明

ndarrayクラスは1次元の場合はベクトル、2次元の場合は行列、3次元以上はテンソルと呼ばれます。

