# NumPy Arrayの条件フィルターの使い方

ndarrayの中の要素の操作


```python
import numpy as np
```

### np.clip(a, a_min, a_max)

最小値と最大値を設定します。<br>その範囲以外の数字は最小値、もしくは最大値を適用します。


```python
array = np.arange(0, 10)
array
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
# 最小値と最大値を設定する．その範囲以外の数字は最小値，もしくは最大値を適用
np.clip(array, 3, 7)
```




    array([3, 3, 3, 3, 4, 5, 6, 7, 7, 7])



ある条件下で最大値と最小値が決定していてその周辺の値が集中するときに何かの値で統一したいときに使用します。

## ndarrayの状態

### np.where()

条件に一致する要素を入れ替えます。 (この場合Trueは1、 Falseは0)

```python
np.where(array > 3, a, b)
```
- arrayが3より大きいなら`a`
- arrayが3以下なら`b`


```python
array = np.arange(10)
array
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
# 条件に一致する要素を入れ替える (この場合Trueは1, Falseは0)
np.where(array > 3, 1, 0)
```




    array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])




```python
np.where(array > 3)
```




    (array([4, 5, 6, 7, 8, 9]),)



##### np.whereの結果を最初のarrayを取得するとき

tupleで結果が返ってくるのでunpackして取得します。


```python
result,  = np.where(array > 3)
print(result)
print(result[0])
```

    [4 5 6 7 8 9]
    4


##### filterするとき


```python
# fillter
array >3result
```




    array([False, False, False, False,  True,  True,  True,  True,  True,
            True])



filterは以下のように実行するのが一般的です。


```python
array[array > 3]
```




    array([4, 5, 6, 7, 8, 9])




```python
# ndarrayでも，flatになってかえってくるので注意
ndarray = array.reshape(2, 5)
print(ndarray.shape)
ndarray[ndarray > 3]
```

    (2, 5)





    array([4, 5, 6, 7, 8, 9])



もともとの形は崩れるので注意が必要です。


```python
ndarray
```




    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])




```python
ndarray > 3
```




    array([[False, False, False, False,  True],
           [ True,  True,  True,  True,  True]])



#### filter操作


```python
# ndarrayのすべての要素が条件を満たすか
# 全て？
(ndarray > 3).all()
```




    False




```python
# １つでも？
(ndarray > 3).any()
```




    True




```python
# axis=0: 列ごとに評価, axis=1: 行ごとに評価
(ndarray > 3).all(axis=0)
```




    array([False, False, False, False,  True])



### 重複を除く np.unique()

ndarayの重複を除きます


```python
array = np.array([1, 1, 2, 3, 4, 2, 3, 4, 5])
# 重複を除く
np.unique(array)
```




    array([1, 2, 3, 4, 5])




```python
# return_counts=Trueで，各要素のカウントを返す
np.unique(array, return_counts=True)
```




    (array([1, 2, 3, 4, 5]), array([2, 2, 2, 2, 1]))




```python
# 結果はtupleなのでunpackして取得
unique_array, unique_cnt = np.unique(array, return_counts=True)

print(unique_array)
print(unique_cnt)
```

    [1 2 3 4 5]
    [2 2 2 2 1]


### 0から連番のカウント数を返す np.bincount()


```python
#0, 1, 2, 3...の各カウント数を返す
np.bincount(array)
```




    array([0, 2, 2, 2, 2, 1])


