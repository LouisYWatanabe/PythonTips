# ユニバーサル関数


### 説明

<b style='color: #AA0000'>ユニバーサル関数</b>とは`ndarray配列`の<b>各要素に対して演算</b>した結果を返す関数のことです。要素ごとの計算なので多次元配列でも用いることができます。ユニバーサル関数には引数が1つのものと2つのものがあります。下記が代表例です。<br>

<b>引数が1つの関数</b><br>
- 要素の絶対値を返す`np.abs()`
- 要素の $e$ (自然対数の底)のべき乗を返す`np.exp()`
- 要素の平方根を返す`np.sqrt()`
など<br>

<b>引数が2つの関数</b><br>
- 要素同士の和を返す`np.add()`
- 要素同士の差を返す`np.subtract()`
- 要素同士の最大値を格納した配列を返す`np.maximum()`
など<br>


### 例

```python
import numpy as np

arr = np.array([8, -27, 32, -8, 16])
print(arr)

# 変数arrの各要素の絶対値を変数arr_absに代入してください
arr_abs = np.abs(arr)
print(arr_abs)

# 変数arr_absの各要素のeのべき乗と平方根を出力してください
print(np.exp(arr_abs))
print(np.sqrt(arr_abs))
```
```python

[  8 -27  32  -8  16]
[ 8 27 32  8 16]
[2.98095799e+03 5.32048241e+11 7.89629602e+13 2.98095799e+03
 8.88611052e+06]
[2.82842712 5.19615242 5.65685425 2.82842712 4.        ]
```
