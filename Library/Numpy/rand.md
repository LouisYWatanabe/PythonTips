# 乱数

```python
import numpy as np
# 0以上1未満の一様乱数
np.random.rand()
# x以上y未満の整数をz個生成する
np.random.randint(x, y, z)
# ガウス分布に従う乱数を生成する
np.random.normal()
```

### 書式


### 引数


### 例

```python
import numpy as np

# randint()をnp.randomと記述しなくてもいいようにimportしてください
from numpy.random import randint

# 乱数を一定のルールで作るように設定（正誤判定用）
np.random.seed(2)
# 変数arr_randintに各要素が1以上11以内の整数の行列(6 × 2)を代入してください
arr_randint = randint(1, 12, (6, 2))
print(arr_randint)

# 変数arr_randomに0以上1未満の一様乱数を3つ代入してください
arr_random = np.random.rand(3)
print(arr_random)
```
```python
[[ 9  9]
 [ 7  3]
 [ 9  8]
 [ 3  2]
 [ 6 11]
 [ 5  5]]
[0.52914209 0.13457995 0.51357812]

```


### np.random.randn()

標準正規分布(平均0, 分散1)からランダムの値が返されます。


```python
# 標準正規分布(平均0, 分散1)からランダムの値が返される
np.random.randn(3, 3)
```




    array([[-0.40128759, -2.76858053,  0.26440809],
           [ 1.46436716, -1.12907022, -0.01169512],
           [-0.85946936, -0.9925092 ,  0.23171985]])




```python
np.random.randn(3, 3, 4)
```




    array([[[-9.30314235e-01,  1.35961166e+00,  1.00640977e+00,
              1.12883182e-01],
            [ 4.33357324e-01, -3.47501864e-01,  3.22368206e-02,
             -1.47281886e+00],
            [ 3.14668130e+00, -2.59372505e+00, -1.02386832e-01,
             -4.10756475e-01]],
    
           [[-2.04697063e+00,  7.71671702e-01, -7.82649950e-01,
             -3.09659776e-01],
            [ 6.24139756e-01, -5.83213105e-01,  2.36357971e-04,
              1.80571839e+00],
            [ 7.10093938e-02, -1.03967692e+00, -7.88069514e-01,
              3.71098513e-01]],
    
           [[ 9.00313247e-01,  7.48414516e-01,  2.66486009e-01,
              9.21602371e-01],
            [ 7.24119106e-01,  1.10496445e-02, -9.24581182e-01,
             -6.78519177e-02],
            [-5.89930053e-01,  1.07458245e+00, -1.51136508e-01,
             -1.91369018e+00]]])



#### 任意の正規分布を使いたい場合 np.random.normal()


```python
# 任意の正規分布を使いたい場合はnp.random.normalを使う
mu = 0
sigma = 1
np.random.normal(mu, sigma)
```




    -0.016068903618148167



### np.random.randint(low, high=None, size=None)

ランダムな整数を取得する


```python
# low以上high未満のランダムな整数でsizeのndarrayを生成
size = (2, 3)
np.random.randint(10, 100, size=size)
```




    array([[45, 34, 38],
           [40, 32, 71]])




```python
np.random.seed(42)
# high=Noneの場合は0以上low未満
np.random.randint(10)
```




    6



### np.random.choice()

リストからランダムな値を取得


```python
index_pool = np.arange(0, 10, 2)
index_pool
```




    array([0, 2, 4, 6, 8])




```python
# 指定したndarrayからランダムで値を取ってくる
np.random.choice(index_pool)
```




    2


### 説明

NumPyは<b style='color: #AA0000'>np.randomモジュール</b>で乱数を生成することができます。<br>

代表的な関数は以下の通りです。
- 0以上1未満の一様乱数を生成する`np.random.rand()`<br>
()の中に指定した整数の個数分の乱数を生成します。<br><br>

- x以上y未満の整数をz個生成する`np.random.randint(x, y, z)`<br>
x以上y未満の整数を生成することに注意しましょう。さらに、zに`(2,3)`のように引数を指定すると、$2×3$の行列を生成することができます。<br><br>

- ガウス分布に従う乱数を生成する`np.random.normal()`<br>
などがあります。

通常これらの関数を用いる時、`np.random.randint()`のように記述しますが、毎回`np.random`を記述するのは手間なだけでなく、コードも読み難くなります。<br>
そこでインポートする際に`from numpy.random import randint`と最初に記述しておくと`randint()`のみで用いることができるようになります。<br>
これは他の関数も同様で、<b>`「from モジュール名 import そのモジュールの中にある関数名」`</b>とすることでモジュール内の関数を直接インポートするので、モジュール名を記述する必要がなくなります。

