# 統計量


```python
import numpy as np
```


```python
# 標準正規分布平均0標準偏差1から乱数生成
std_norm = np.random.randn(5, 5)
std_norm
```




    array([[-1.33082549e+00, -1.18390005e+00, -1.63746216e-01,
            -5.81639577e-01, -4.10024873e-01],
           [ 7.73292404e-01, -1.31170698e-03, -1.23818417e+00,
             2.65935872e+00,  5.89983217e-02],
           [ 1.80424381e-01,  8.11724984e-01, -1.72048633e+00,
            -1.71462392e+00, -1.43287850e+00],
           [ 5.67541602e-01, -1.01007869e+00, -1.53243492e+00,
             5.18809699e-01, -4.02134364e-01],
           [ 1.92172076e-01, -1.33430761e+00, -7.96193913e-02,
            -9.97761622e-01,  8.37023958e-02]])



### 最大値


```python
#最大値を取得
std_norm.max()
```




    2.6593587227253344




```python
np.max(std_norm)
```




    2.6593587227253344



最大値のindexを取得


```python
#最大値のindexを取得
std_norm.argmax()
```




    8




```python
std_norm.flatten()[8]
```




    2.6593587227253344



### 最小値


```python
#最小値を取得
std_norm.min()
```




    -1.7204863338052878



最小値のindexを取得


```python
#最小値のindexを取得
std_norm.argmin()
```




    12




```python
std_norm.flatten()[12]
```




    -1.7204863338052878



## 平均


```python
std_norm.mean()
```




    -0.3715173140265549



## 中央値


```python
#中央値
np.median(std_norm)
```




    -0.4021343638842082



中央値は平均よりも時間がかかります


```python
# medianはmeanよりも時間がかかる
# timeモジュールを使って処理時間を計測する
import time
big = np.random.randint(10, 100, (1000, 10000))
before = time.time()
np.median(big)
after_median = time.time()
print('median took {:.2f} sec'.format(after_median - before))
np.mean(big)
after_mean = time.time()
print('mean took {:.2f} sec'.format(after_mean - after_median))
```

    median took 0.71 sec
    mean took 0.01 sec


## 標準偏差

平均の差を2乗した合計を、データ数で割った正の平方根


```python
#標準偏差 (standard deviation)
std_norm.std()
```




    1.00108847109738



### 行ごと列ごとに統計量を求める

axis引数で特定のaxisにて統計量を計算(axis=0: 列, axis=1: 行)



```python
# axis引数で特定のaxisにて統計量を計算(axis=0: 列, axis=1: 行)
print(std_norm)
print()
print('max value on axis=0: {}'.format(std_norm.max(axis=0)))
print('max value on axis=1: {}'.format(std_norm.max(axis=1)))
```

    [[-1.33082549e+00 -1.18390005e+00 -1.63746216e-01 -5.81639577e-01
      -4.10024873e-01]
     [ 7.73292404e-01 -1.31170698e-03 -1.23818417e+00  2.65935872e+00
       5.89983217e-02]
     [ 1.80424381e-01  8.11724984e-01 -1.72048633e+00 -1.71462392e+00
      -1.43287850e+00]
     [ 5.67541602e-01 -1.01007869e+00 -1.53243492e+00  5.18809699e-01
      -4.02134364e-01]
     [ 1.92172076e-01 -1.33430761e+00 -7.96193913e-02 -9.97761622e-01
       8.37023958e-02]]
    
    max value on axis=0: [ 0.7732924   0.81172498 -0.07961939  2.65935872  0.0837024 ]
    max value on axis=1: [-0.16374622  2.65935872  0.81172498  0.5675416   0.19217208]

