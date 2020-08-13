# 結合と転置


```python
import numpy as np
```

### np.concatenate()

ndarrayを連結します


```python
ndarray_even = np.arange(0, 18, 2).reshape(3, 3)
ndarray_even
```




    array([[ 0,  2,  4],
           [ 6,  8, 10],
           [12, 14, 16]])




```python
ndarray_odd = np.arange(1, 19, 2).reshape(3, 3)
ndarray_odd
```




    array([[ 1,  3,  5],
           [ 7,  9, 11],
           [13, 15, 17]])




```python
# concatenate: arrayを連結する
np.concatenate([ndarray_even, ndarray_odd])
```




    array([[ 0,  2,  4],
           [ 6,  8, 10],
           [12, 14, 16],
           [ 1,  3,  5],
           [ 7,  9, 11],
           [13, 15, 17]])




```python
# 既存のaxisを指定して特定の軸で連結する．デフォルトは0
np.concatenate([ndarray_even, ndarray_odd], axis=1)
```




    array([[ 0,  2,  4,  1,  3,  5],
           [ 6,  8, 10,  7,  9, 11],
           [12, 14, 16, 13, 15, 17]])



### np.stack()




```python
# concatenateと異なり，新たなaxisを作成　（デフォルトaxis=0）
stacked_array = np.stack([ndarray_even, ndarray_odd])
stacked_array
```




    array([[[ 0,  2,  4],
            [ 6,  8, 10],
            [12, 14, 16]],
    
           [[ 1,  3,  5],
            [ 7,  9, 11],
            [13, 15, 17]]])




```python
print('stacked_array shape: {}'.format(stacked_array.shape))
```

    stacked_array shape: (2, 3, 3)



```python
np.stack([ndarray_even, ndarray_odd], axis=1)
```




    array([[[ 0,  2,  4],
            [ 1,  3,  5]],
    
           [[ 6,  8, 10],
            [ 7,  9, 11]],
    
           [[12, 14, 16],
            [13, 15, 17]]])




```python
st_ar = _
print('stacked_array shape: {}'.format(st_ar.shape))
```

    stacked_array shape: (3, 2, 3)



```python
np.stack([ndarray_even, ndarray_odd], axis=2)
```




    array([[[ 0,  1],
            [ 2,  3],
            [ 4,  5]],
    
           [[ 6,  7],
            [ 8,  9],
            [10, 11]],
    
           [[12, 13],
            [14, 15],
            [16, 17]]])




```python
st_ar = _
print('stacked_array shape: {}'.format(st_ar.shape))
```

    stacked_array shape: (3, 3, 2)


これは`axis=-1`を設定したときと同じ動作です。


```python
st_ar = np.stack([ndarray_even, ndarray_odd], axis=-1)
print('stacked_array shape: {}'.format(st_ar.shape))
```

    stacked_array shape: (3, 3, 2)


この操作は画像処理の奥行きを設定するときによく使用します。

## np.transposeと.T


```python
ndarray = np.random.randn(3, 4)
print(ndarray.shape)
ndarray
```

    (3, 4)





    array([[ 0.6192402 , -1.3983397 ,  0.40011583,  1.12127087],
           [-1.24141052,  0.51583727,  0.79414423,  0.74182309],
           [-0.243559  ,  4.11196662, -1.03036215, -0.0971766 ]])




```python
# 転置 (transpose)
transpose_ndarray = np.transpose(ndarray)
transpose_ndarray
```




    array([[ 0.6192402 , -1.24141052, -0.243559  ],
           [-1.3983397 ,  0.51583727,  4.11196662],
           [ 0.40011583,  0.79414423, -1.03036215],
           [ 1.12127087,  0.74182309, -0.0971766 ]])




```python
transpose_ndarray.shape
```




    (4, 3)




```python
# .Tが便利
ndarray.T
```




    array([[ 0.6192402 , -1.24141052, -0.243559  ],
           [-1.3983397 ,  0.51583727,  4.11196662],
           [ 0.40011583,  0.79414423, -1.03036215],
           [ 1.12127087,  0.74182309, -0.0971766 ]])




```python
# n-dimentionalでも同じ
ndarray = np.random.randn(3, 4, 5)
```


```python
np.transpose(ndarray).shape
```




    (5, 4, 3)




```python

```
