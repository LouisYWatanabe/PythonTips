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

