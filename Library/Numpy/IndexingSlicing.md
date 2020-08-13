# IndexingとSlicing


```python
import numpy as np
```

## Indexingで要素取得

### 1次元配列


```python
# 一列の場合は簡単
ndarray = np.array([1, 2, 3, 4])
# indexは0から
print(ndarray[0])
# 最後は-1
print(ndarray[-1])
```

    1
    4


### 多次元配列


```python
# N-dimentional
ndarray = np.array([[1, 2], [3, 4], [5, 6]])
#まずは2段階で考える
# [0]で最初の要素[1, 2]を取得
print(ndarray[0])
# それに対してさらにindexing
print(ndarray[0][1])
# ndarrayでは以下のようにまとめて記述するのが一般的
print(ndarray[0, 1]) #[一個目のindex, 二個目のindex]
#　画像の場合は(height, width, channel)の並びになる
```

    [1 2]
    2
    2


## 複数の要素の取得Slicing

[N:M] N以上M未満を返す

### 1次元配列


```python
# 一列の場合
ndarray = np.array([1, 2, 3, 4])
# [N:M] N以上M未満を返す
print(ndarray[1:3]) #[2, 3]
# Nを省略すると最初からM未満
print(ndarray[:3]) #[1, 2, 3]
# Mを省略するとNから最後の要素まで
print(ndarray[1:]) #[2, 3, 4]
# 最後の要素はindex=-1なのでこのようにすれば'最後から〇〇番目の要素まで'
print(ndarray[:-2]) #[1, 2]
# もしくは'後ろ〇番目の要素から'
print(ndarray[-2:]) #[3, 4]
# ':'のみだと，全ての要素を取得
print(ndarray[:]) # [1, 2, 3, 4]
```

    [2 3]
    [1 2 3]
    [2 3 4]
    [1 2]
    [3 4]
    [1 2 3 4]


### 多次元配列


```python
array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
array
```




    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12],
           [13, 14, 15, 16]])




```python
#　[行, 列]
#　まずは１つ目
array[:2]
```




    array([[1, 2, 3, 4],
           [5, 6, 7, 8]])




```python
# からの２つ目
array[:2, 1:]
```




    array([[2, 3, 4],
           [6, 7, 8]])




```python
# これとは同じにならないことに注意
array[:2][1:]
```




    array([[5, 6, 7, 8]])




```python
# 列だけ抽出することも可能
array[:, 2]
```




    array([ 3,  7, 11, 15])


