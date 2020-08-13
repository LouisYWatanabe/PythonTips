# 要素を指定して行列生成 (zeros, ones, eye)


```python
import numpy as np
```

## np.zeros(shape)

要素がすべて`0`の`ndarray`


```python
# 要素が全て０のndarrayを作成
shape = (3, 3)
np.zeros(shape)
```




    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])




```python
# tupleではなくintをいれると，一列のarrayができる
np.zeros(3)
```




    array([0., 0., 0.])




```python
np.zeros((5, 4)) + 5
```




    array([[5., 5., 5., 5.],
           [5., 5., 5., 5.],
           [5., 5., 5., 5.],
           [5., 5., 5., 5.],
           [5., 5., 5., 5.]])



## np.ones(shape)

要素がすべて`1`の`ndarray`


```python
# np.zerosの「1」版　要素が全て１のndarrayを作成
shape = (3, 3)
np.ones(shape)
```




    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])




```python
np.ones(3)
```




    array([1., 1., 1.])




```python
np.ones((5, 4)) * 5
```




    array([[5., 5., 5., 5.],
           [5., 5., 5., 5.],
           [5., 5., 5., 5.],
           [5., 5., 5., 5.],
           [5., 5., 5., 5.]])



## np.eye(N)

対角要素がすべて`1`の`ndarray`


```python
# N x Nの単位行列を作成
#単位行列：対角成分が全て１の正方行列
np.eye(3)
```




    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])




```python
# N行M列の行列も作れる
np.eye(3, 4)
```




    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.]])


