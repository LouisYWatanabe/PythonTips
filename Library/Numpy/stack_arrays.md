# 多次元配列の宣言


```python
import numpy as np
# 2次元配列を転置で宣言
array = np.array([1, 2, 3, 4, 5, 6], ndmin=2).T
```

### 書式

	np.array(配列)

### 引数

	ndmin=配列の次元数

### 例

```python
import numpy as np

# 2次元配列を転置で宣言
array = np.array([1, 2, 3, 4, 5, 6], ndmin=2).T

print(array)

```
```
[[1]
 [2]
 [3]
 [4]
 [5]
 [6]]
```

```python
import numpy as np

input_list = [1., -1.5, 2.]
inputs = np.array(
    np.append(
        input_list, [1]),        # バイアスのために[1]
    ndmin=2,                     # 2次元配列化
).T                              # 転置

inputs
```

```
array([[ 1. ],
       [-1.5],
       [ 2. ],
       [ 1. ]])
```

### 説明
