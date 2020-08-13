# 形状 shape

`.shape`でndarrayの行列の数を確認することができます。


```python
import numpy as np
```


```python
ndarray = np.array([[1, 2], [3, 4], [5, 6]])
# .shapeでshapeを確認
ndarray.shape
```




    (3, 2)



3行2列であることが分かります。

# reshape : shapeの変更


```python
# reshapeでshapeを変更できる
ndarray.reshape(2, 3)
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
# 以下の２つは似てるようでshapeの結果が異なるので注意
ndarray1 = np.array([1, 2, 3])
ndarray2 = np.array([[1, 2, 3]])
print('result is vector {}'.format(ndarray1.shape))
print('result is queye {}'.format(ndarray2.shape))

```

    result is vector (3,)
    result is queye (1, 3)


1行目の結果はベクトル<br>2行目の結果は行列です。

## 次元の追加

`np.expand_dims(ndarray, axis)`でndarrayの次元を増やすことができます。


```python
# 1行目の次元を追加します
np.expand_dims(ndarray1, axis=0)
```




    array([[1, 2, 3]])



`axis=-1`とすることで最後の次元が追加されます。


```python
# axis=-1にすると最後のrankを追加
# (3,) -> (3, 1)
expand_ndarray = np.expand_dims(ndarray1, axis=-1)
expand_ndarray.shape
```




    (3, 1)



## 次元の削減

`np.squeeze(ndarray)`で次元を削減することができます。


```python
expand_ndarray.shape
```




    (3, 1)




```python
np.squeeze(expand_ndarray)
```




    array([1, 2, 3])



## arrayを1次元にする

`np.flatten()`でarrayを1次元にすることができます


```python
ndarray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
ndarray
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
ndarray.shape
```




    (3, 3)




```python
ndarray.flatten()
```




    array([1, 2, 3, 4, 5, 6, 7, 8, 9])


