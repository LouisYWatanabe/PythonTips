# ブールインデックス参照


```python
arr = np.array([2, 4, 6, 7]) 
print(arr[np.array([True, True, True, False])])

>>> 出力結果
[2 4 6]
```

### 書式

	arr[ndarrayの論理値の配列]

### 引数

- <b>arr[ndarrayの論理値の配列]</b>
論理値配列のTrueに該当する箇所の要素のndarray配列を作成して返します。

### 例

```python
import numpy as np

arr = np.array([2, 3, 4, 5, 6, 7])

# 変数arrの各要素が2で割り切れるかどうかを示す真偽値の配列を出力してください
print(arr % 2 == 0)

# 変数arr各要素のうち2で割り切れる要素の配列を出力してください
print(arr[arr % 2 == 0])
```
```python
[ True False  True False  True False]
[2 4 6]
```

### 説明

<b style='color: #AA0000'>ブールインデックス参照</b>とは、<b>論理値(`True/False`)の配列を用いて要素を取り出す方法</b>のことです。`arr[ndarrayの論理値の配列]`と記述すると、論理値配列の`True`に該当する箇所の要素の`ndarray配列`を作成して返します。

