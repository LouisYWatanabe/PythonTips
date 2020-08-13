# 数値のArrayを関数で作成(arange, linspace, logspace)


```python
import numpy as np
```

## np.arange(start=0, stop, step=1)


```python
# [start, start+step, start+2*step, ..., stop未満]
# stopとstepは省略可能．その場合start=0, step=1が入る
np.arange(5) # np.arange(0, 5, 1)と同じ
```




    array([0, 1, 2, 3, 4])




```python
# 1~10までstep=2で取得
np.arange(1, 10, 2)
```




    array([1, 3, 5, 7, 9])




```python
# stepを負の数にすれば降順も可能
np.arange(10, 1, -1)
```




    array([10,  9,  8,  7,  6,  5,  4,  3,  2])



## np.linspace(start, stop, num=50)

start以上stop以下の数をnum個で区切った値のリストにして取得します。<br>取り出したい数が分かっているときに使用します。


```python
np.linspace(0, 10, 5)
```




    array([ 0. ,  2.5,  5. ,  7.5, 10. ])




```python
# start以上stop以下の数をnum個で区切った値のリスト(linear)
# stopを含むので注意
np.linspace(0, 10, 11)
```




    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])



## np.logspace(start, stop, num=50)

10**start以上10**stop以下の数をnum個で区切った数のリスト(logarithm)で取得します。<br>コード例では10^0から10^3を10個で区切った数として出力しています。


```python
# 10**start以上10**stop以下の数をnum個で区切った数のリスト(logarithm)
np.logspace(0, 3, 10)
```




    array([   1.        ,    2.15443469,    4.64158883,   10.        ,
             21.5443469 ,   46.41588834,  100.        ,  215.443469  ,
            464.15888336, 1000.        ])


