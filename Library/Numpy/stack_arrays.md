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

## 多次元配列の作成


```python
import numpy as np
```


```python
# 二次元配列の作成
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix)
```

    [[1 2 3]
     [4 5 6]
     [7 8 9]]



```python
# indexingにより要素を取ってくる
matrix[0][0]
```




    1




```python
matrix[0]
```




    array([1, 2, 3])



#### データタイプ NumPyのデータタイプ (dtype)



```python
# 各要素にはdtypeオブジェクトが入っている
type(matrix[0][0])
```




    numpy.int64



要素のデータタイプを指定することも可能です。


```python
matrix = np.array(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    dtype=complex    # 複素数
)
type(matrix[0][0])
```




    numpy.complex128




```python
matrix
```




    array([[1.+0.j, 2.+0.j, 3.+0.j],
           [4.+0.j, 5.+0.j, 6.+0.j],
           [7.+0.j, 8.+0.j, 9.+0.j]])




```python
# unsigned int 8bit 256まで
ndarray = np.array([1, 2, 3], dtype=np.uint8)
ndarray
```




    array([1, 2, 3], dtype=uint8)



`float32`は機械学習に使うデータを保存するときによく使用します。


```python
ndarray = np.array([1, 2, 3], dtype=np.float32)
ndarray
```




    array([1., 2., 3.], dtype=float32)



`float64`は実際のモデルの学習に使用することが多いです。


```python
ndarray = np.array([1, 2, 3], dtype=np.float64)
ndarray
```




    array([1., 2., 3.])

