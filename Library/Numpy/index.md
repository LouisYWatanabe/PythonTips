# インデックス参照とスライス


```python
import numpy as np

arr = np.arange(10)
print(arr)

# 変数arrの要素の内3, 4, 5だけを出力してください
print(arr[3:6])

# 変数arrの要素の内3, 4, 5を24に変更してください
arr[3:6] = 24
print(arr)
```

### 書式

	arr[start:end]

### 引数

- <b>[start:end]</b>
startから(end-1)までのリストが作成されます。

### 例

```python
import numpy as np

arr = np.arange(10)
print(arr)

# 変数arrの要素の内3, 4, 5だけを出力してください
print(arr[3:6])

# 変数arrの要素の内3, 4, 5を24に変更してください
arr[3:6] = 24
print(arr)
```
```python
[0 1 2 3 4 5 6 7 8 9]
[3 4 5]
[ 0  1  2 24 24 24  6  7  8  9]
```

### 説明

ndarray配列はPythonのリストと同じように<b>代入先の変数の値を変更すると元のndarray配列の値も変更</b>されます。<br>
そのため、ndarray配列をコピーして別の変数として扱いたい場合は、`copy( )`メソッドを使用します。このメソッドは`コピーしたい配列.copy()`で使用できます。

