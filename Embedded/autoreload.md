# Pythonスクリプトを別ファイルに保存してJupyterで読み込む

`util.py`というファイルを同じフォルダ内に以下の内容で作成し、このjupyterで読み込んでみます。

```python
def multiply(a, b):
    return a * b
```


```python
# util.pyファイルのmultiply()関数を読み込む
from util import multiply
```

`util.py`がフォルダの中にある場合は

```python
from フォルダ名 import util
```

と宣言することで読み込むことができます。


```python
multiply(2, 3)
```




    6



`util.py`ファイルを以下のように変更して'multiply2()'関数を実行してみます

```python
def multiply2(a, b):
    print('{} * {} = {}'.format(a, b, a*b))
    return a * b
```


```python
multiply2(2, 3)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-3-c5ba07743ce4> in <module>
    ----> 1 multiply2(2, 3)
    

    NameError: name 'multiply2' is not defined



```python
multiply(2, 3)
```




    6



別ファイルから読み込んだ関数そのものを変更してもその変更は反映されません。<br>変更を反映したい場合は、関数を読み込むときにマジックコマンド`%load_ext autoloade`と`%autoreload 2`(0: オートリロードしない、1: このセルだけリロードする、2: 常にリロードする)

`util.py`ファイルを以下のように戻して常にリロードする設定にし、<br>先ほどの処理を再度実行します。

```python
def multiply(a, b):
    return a * b
```


```python
# 変更したモジュールが反映されるようにする
# 'autoreload'というextensionをloadする
%load_ext autoreload
#　スクリプトを実行するたびに毎回reloadするように設定
%autoreload 2
import util
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload



```python
util.multiply(2, 3)
```




    6



`util.py`ファイルを以下のように変更して'multiply2()'関数を実行してみます

```python
def multiply2(a, b):
    print('{} * {} = {}'.format(a, b, a*b))
    return a * b
```


```python
util.multiply2(2, 3)
```

    2 * 3 = 6





    6


