# Datatype



## Numbers

数字の型を確認します


```python
# 型の確認にはtype()を使用します。
type(3)
```




    int



3はint型であることが分かります


```python
type(3.3)
```




    float




```python
type(3 + 3)
```




    int




```python
type(3 + 3.3)
```




    float



intとfloatで計算すると結果はfloatになります。<br>つまり、型の違う計算の結果は情報量の多いDataTypeになるということです。

#### == そのオブジェクトは同じかどうかの判定


```python
1 == 1.0
```




    True




```python
2.0 == 2.1
```




    False



#### 10の-a乗の表記


```python
1e-3    # 0.001
```




    0.001


