# 1次元配列の宣言


```python
import numpy as np
# 2次元でいうと、水平方向に(horizontal)連結します
np.hstack(tup)
# 縦方向に連結します
np.vstack(tup)
```

### 書式

	np.hstack(tup)

	np.vstack(tup)

### 引数

- <b>tup</b>
	結合したい配列(ndarray)を指定します。
数値、または文字列のリストを代入する

### 例

```python
import numpy as np

a = np.arange(12)
b = np.arange(2)

print( np.hstack((a, b)) )


```
```python
[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,  0,  1]
```

```python
import numpy as np

a = np.arange(12).reshape(-1,1) # 12個の要素を持つ縦ベクトル
b = np.arange(2).reshape(-1,1) # 2個の要素を持つ縦ベクトル
print(a)
print()
print(b)
print()
print( np.vstack((a, b)) )
```
```python
   [[ 0],
	[ 1],
	[ 2],
	[ 3],
	[ 4],
	[ 5],
	[ 6],
	[ 7],
	[ 8],
	[ 9],
	[10],
	[11]]

   [[ 0],
	[ 1]]

   [[ 0],
	[ 1],
	[ 2],
	[ 3],
	[ 4],
	[ 5],
	[ 6],
	[ 7],
	[ 8],
	[ 9],
	[10],
	[11],
	[ 0],
	[ 1]]
```
### 説明

`np.hstack`で水平方向に連結。
`np.vstack`で縦ベクトルの連結。

