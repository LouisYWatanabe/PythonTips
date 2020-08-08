# 1次元配列の計算


```python
import numpy as np
storages = np.array([1, 2, 3, 4])
storages += storages
```

### 書式



### 引数


### 例

```python
import numpy as np

arr = np.array([2, 5, 3, 4, 8])

# arr + arr
print('arr + arr')
print(arr + arr)
print() # 空行を出力

# arr - arr
print('arr - arr')
print(arr - arr)
print() # 空行を出力

# arr ** 3
print('arr ** 3')
print(arr ** 3)
print() # 空行を出力

# 1 / arr
print('1 / arr')
print(1 / arr)
print() # 空行を出力

```
```python
arr + arr
[ 4 10  6  8 16]

arr - arr
[0 0 0 0 0]

arr ** 3
[  8 125  27  64 512]

1 / arr
[0.5        0.2        0.33333333 0.25       0.125     ]
```

### 説明

リストでは、要素ごとの計算を行うためには<b>ループさせて要素を一つずつ取り出して足し算</b>を行う必要がありましたが、`ndarray配列`ではループさせる必要はありません。`ndarray配列`同士の算術演算では、<b>同じ位置にある要素同士</b>が計算されます。