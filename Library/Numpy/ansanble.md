# ユニバーサル関数


### 説明
Numpyの<b style='color: #AA0000'>集合関数</b>は数学の集合演算を行う関数のことで、1次元配列のみを対象としています。<br>

代表的な関数は、以下の通りです。
- 配列要素から重複を取り除きソートした結果を返す`np.unique()`
- 配列xとyのうち少なくとも一方に存在する要素を取り出しソートする`np.union1d(x, y)`（和集合）
- 配列xとyのうち共通する要素を取り出しソートする`np.intersect1d(x, y)`（積集合）
- 配列xと配列yに共通する要素を配列xから取り除きソートする`np.setdiff1d(x, y)`（差集合）
など

### 例

```python
import numpy as np

array1 = [3, 5, 4, 9, 8, 2]
array2 = [7, 4, 3, 1, 9]

# unique()を用いて、変数array1の重複をなくした配列を変数array_unique1に代入してください
array_unique1 = np.unique(array1)
print(array_unique1)

# 変数array_unique1と変数array2の和集合を出力してください
print(np.union1d(array_unique1, array2))

# 変数array_unique1と変数array2の積集合を出力してください
print(np.intersect1d(array_unique1, array2))

# 変数array_unique1から変数array2を引いた差集合を出力してください
print(np.setdiff1d(array_unique1, array2))
```
```python
[2 3 4 5 8 9]
[1 2 3 4 5 7 8 9]
[3 4 9]
[2 5 8]
```
