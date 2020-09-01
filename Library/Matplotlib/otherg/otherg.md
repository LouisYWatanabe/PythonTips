```python
# plt.show()で可視化されない人はこのセルを実行してください。
%matplotlib inline
```

# 様々なグラフを作る

- **[折れ線グラフ](#折れ線グラフ)**
    - **[マーカーの種類と色を設定する](#マーカーの種類と色を設定する)**
    - **[線のスタイルと色を設定する](#線のスタイルと色を設定する)**
<br><br>
- **[棒グラフ](#棒グラフ)**
    - **[棒グラフを作成する](#棒グラフを作成する)**
    - **[横軸にラベルを設定する](#横軸にラベルを設定する)**
    - **[積み上げ棒グラフを作成する](#積み上げ棒グラフを作成する)**
<br><br>
- **[ヒストグラム](#ヒストグラム)**
    - **[ヒストグラムを作成する](#3.3.1-ヒストグラムを作成する)**
    - **[ビン数を設定する](#ビン数を設定する)**
    - **[正規化を行う](#正規化を行う)**
    - **[累積ヒストグラムを作成する](#累積ヒストグラムを作成する)**
<br><br>
- **[散布図](#散布図)**
    - **[散布図を作成する](#散布図を作成する)**
    - **[マーカーの種類と色を設定する](#マーカーの種類と色を設定する)**
    - **[値に応じてマーカーの大きさを設定する](#値に応じてマーカーの大きさを設定する)**
    - **[値に応じてマーカーの濃さを設定する](#値に応じてマーカーの濃さを設定する)**
    - **[カラーバーを表示する](#カラーバーを表示する)**
<br><br>
- **[円グラフ](#円グラフ)**
    - **[円グラフを作成する](#円グラフを作成する)**
    - **[円グラフにラベルを設定する](#円グラフにラベルを設定する)**
    - **[特定の要素を目立たせる](#特定の要素を目立たせる)**
<br><br>
- **[3Dグラフ](#3Dグラフ)**
    - **[3D Axesを作成する](#3D-Axesを作成する)**
    - **[曲面を作成する](#曲面を作成する)**
    - **[3Dヒストグラムを作成する](#3Dヒストグラムを作成する)**
    - **[3D散布図を作成する](#3D散布図を作成する)**
    - **[3Dグラフにカラーマップを適用する](#3Dグラフにカラーマップを適用する)**

***

## 折れ線グラフ

### マーカーの種類と色を設定する

<b>折れ線グラフ</b>は`matplotlib.pyplot.plot()`を用いて描画します。
横軸のデータ`x`、縦軸のデータ`y`に加え、`marker="指定子"`を指定すると<b>マーカーの種類（形）を設定でき</b>、`markerfacecolor="指定子"`を指定すると<b>マーカーの色を設定できます</b>。<br>
```Python
matplotlib.pyplot.plot(x, y, marker="マーカーの種類", markerfacecolor="マーカーの色")
```
以下は指定できるマーカーの種類とその色の一部です。<br>
<br>
<b>マーカー</b>
- `"o"`: 円
- `"s"`: 四角
- `"p"`: 五角形
- `"*"`: 星
- `"+"`: プラス
- `"D"`: ダイアモンド

<b>色</b>
- `"b"` : 青
- `"g"` : 緑
- `"r"` : 赤
- `"c"` : シアン
- `"m"` : マゼンタ
- `"y"` : 黄色
- `"k"` : 黒
- `"w"` : 白

#### 例

- 赤色の円マーカーを用いて折れ線グラフを作成する

x軸に対応するデータは変数`days`、y軸に対応するデータは変数`weight`が用意されています。
赤色は `"r"` で指定します


```python
import numpy as np
import matplotlib.pyplot as plt

days = np.arange(1, 11)
weight = np.array([10, 14, 18, 20, 18, 16, 17, 18, 20, 17])

# 表示の設定
plt.ylim([0, weight.max()+1])
plt.xlabel("days")
plt.ylabel("weight")

# 円マーカーを赤色でプロットし折れ線グラフを作成
plt.plot(days, weight, marker="o", markerfacecolor="r")

plt.show()
```


![png](output_9_0.png)


### 線のスタイルと色を設定する

`matplotlib.pyplot.plot()`に`linestyle="指定子"`を指定すると<b>線のスタイルを設定でき</b>、`color="指定子"`を指定すると<b>線の色を設定できます</b>。

```Python
matplotlib.pyplot.plot(x, y, linestyle="線のスタイル", color="線の色")
```
以下は指定できる線の種類とその色の一部です。<br>
<b>線のスタイル</b>
- `"-"`: 実線
- `"--"`: 破線
- `"-."`: 破線（点入り）
- `":"`: 点線

<b>色</b>
- `"b"` : 青
- `"g"` : 緑
- `"r"` : 赤
- `"c"` : シアン
- `"m"` : マゼンタ
- `"y"` : 黄色
- `"k"` : 黒
- `"w"` : 白

#### 例

- 円マーカーを赤色でプロットし、青の破線の折れ線グラフを作成してください。
x軸に対応するデータは変数`days`、y軸に対応するデータは変数`weight`が用意されています。


```python
import numpy as np
import matplotlib.pyplot as plt

days = np.arange(1, 11)
weight = np.array([10, 14, 18, 20, 18, 16, 17, 18, 20, 17])

# 表示の設定
plt.ylim([0, weight.max()+1])
plt.xlabel("days")
plt.ylabel("weight")

# 円マーカーを赤色でプロットし、青の破線の折れ線グラフを作成
plt.plot(days, weight, linestyle="--", color="b", marker="o", markerfacecolor="r")

plt.show()
```


![png](output_19_0.png)

## 棒グラフ

### 棒グラフを作成する

<b>棒グラフ</b>は`matplotlib.pyplot.bar()`を用いて描画します。
横軸のデータ`x`とこれに対応する縦軸のデータ`y`を指定します。

```Python
matplotlib.pyplot.bar(x, y)
```

#### 例

- 横軸に`x`、縦軸に`y`が対応する棒グラフを作成します。


```python
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6]
y = [12, 41, 32, 36, 21, 17]

# 棒グラフを作成します
plt.bar(x, y)

plt.show()
```


![png](output_30_0.png)


### 横軸にラベルを設定する

棒グラフの横軸にラベルをつける方法は、折れ線グラフやその他のグラフと異なります。
`matplotlib.pyplot.bar()`に`tick_label=[ラベルのリスト]`を指定すると<b>横軸のラベルを設定できます</b>。

```Python
matplotlib.pyplot.bar(x, y, tick_label=[ラベルのリスト])
```

#### 例

- 横軸に`x`、縦軸に`y`のデータが対応する棒グラフを作成し、横軸にラベルを設定してください。
ラベルのリストは変数`labels`が用意されています。


```python
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6]
y = [12, 41, 32, 36, 21, 17]
labels = ["Apple", "Orange", "Banana", "Pineapple", "Kiwifruit", "Strawberry"]

# 棒グラフを作成し、横軸にラベルを設定
plt.bar(x, y, tick_label=labels)

plt.show()
```


![png](output_40_0.png)

### 積み上げ棒グラフを作成する

2系列以上のデータを同じ項目について積み上げて表現したグラフを<b style='color: #AA0000'>積み上げ棒グラフ</b>と呼びます。
`matplotlib.pyplot.bar()`に`bottom=[データ列のリスト]`を指定すると、対応するインデックスで<b>下側に余白を設定できます。</b>
すなわち、2系列目以降をプロットする際に下に表示したい系列を`bottom`に指定することで積み上げ棒グラフが作成できます。

また、`plt.legend("系列1のラベル", "系列2のラベル")` と指定すると凡例を設定できます。
```Python
matplotlib.pyplot.bar(x, y, bottom=[データ列のリスト])
```

#### 例

- 横軸に`x`、縦軸に`y1`、`y2`のデータが対応する積み上げ棒グラフを作成し、横軸にラベルを設定します。
ラベルのリストは 変数`labels`が用意されています。


```python
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6]
y1 = [12, 41, 32, 36, 21, 17]
y2 = [43, 1, 6, 17, 17, 9]
labels = ["Apple", "Orange", "Banana", "Pineapple", "Kiwifruit", "Strawberry"]

# 積み上げ棒グラフを作成し、横軸にラベルを設定
plt.bar(x, y1, tick_label=labels)
plt.bar(x, y2, bottom=y1)

# 系列ラベルの設定が可能
plt.legend(("y1", "y2"))

plt.show()
```


![png](output_50_0.png)


#### ヒント

- `plt.bar()`に`bottom=データ列` と指定すると、対応するインデックスで下側の余白を設定できます。

## ヒストグラム

### ヒストグラムを作成する

データを扱う際には最初に<b>データの全体的な傾向</b>を掴む事が非常に大切です。<br>

たとえば、あるクラスの身長の傾向を把握することを考えてみましょう。各生徒の身長を個々1cm単位で眺めても、全体の傾向は掴めません。身長を10cm単位で区切って各区間の生徒数をカウントすると、全体の傾向を掴めます。
このように、各区間に収まるデータ件数をカウントしたものを<b style='color:#AA0000'>度数分布</b>と呼びます。

<b style='color: #AA0000'>度数分布</b>を可視化する際には`ヒストグラム`という、<b>縦軸に度数（回数）、横軸に階級（範囲）をとった統計グラフ</b>が多く使われます。
ヒストグラムは`matplotlib.pyplot.hist()`を用いて描画します。

```Python
matplotlib.pyplot.hist(リスト型のデータ配列)
```

#### 例

- データ列の変数`data`に入っているデータのヒストグラムを作成してください。


```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
data = np.random.randn(10000)

# データ列の変数dataのヒストグラムを作成
plt.hist(data)

plt.show()
```


![png](output_61_0.png)

### ビン数を設定する

ヒストグラムを作成する際、<b>データをいくつの階級に分けるか</b>が重要になります。
その階級の数を<b style='color: #AA0000'>ビン数</b>と言います。
ビン数を正しく決定することでヒストグラムの特徴を正しく掴むことができます。

`matplotlib.pyplot.hist()`に`bins`を指定すると<b>任意のビン数の階級に分けることができます</b>。`bins="auto"`と指定すると、ビン数が自動で設定されます。

```Python
matplotlib.pyplot.hist(リスト型のデータ列, bins=ビン数)
```

#### 例

- データ列の変数`data`を用いて、ビン数100のヒストグラムを作成します。


```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
data = np.random.randn(10000)

# ビン数100のヒストグラムを作成
plt.hist(data, bins=100)

plt.show()
```


![png](output_71_0.png)

### 正規化を行う

偏差値のツリガネ型のグラフを見た事がある方は多いと思います。あのグラフ（正規分布と言います）は平均が0, 分散が1となるように調整されたグラフで、その結果面積が1となり「偏差値60以上の人は上位15.87%である」等の分布の計算が便利になります。
成績や身長など、自然界の多くのデータは正規分布に近い形になる事が知られています。

ヒストグラムもデータの分布が正規分布であると仮定すると計算が便利になります。ヒストグラムの分布を正規分布と仮定したとき、合計値が1になるようにヒストグラムを操作することを<b style='color: #AA0000'>正規化</b>または<b style='color: #AA0000'>標準化</b>と呼びます。<br>
`matplotlib.pyplot.hist()`に`density=True`を指定すると<b>ヒストグラムの正規化を行えます</b>。
```Python
matplotlib.pyplot.hist(リスト型のデータ列, density=True)
```

#### 例

- データ列の変数`data`を用いて、正規化されたビン数100のヒストグラムを作成


```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
data = np.random.randn(10000)

# 正規化されたビン数100のヒストグラムを作成
plt.hist(data, bins=100, density=True)

plt.show()
```


![png](output_81_0.png)


### 累積ヒストグラムを作成する

度数を全体の割合で表したものを<b style='color: #AA0000'>相対度数</b>と言い、その階級までの相対度数の和を<b style='color: #AA0000'>累積相対度数</b>といいます。<br>
累積相対度数は最終的に1となります。
累積相対度数をヒストグラムで表したものを累積ヒストグラムと呼びます。
累積ヒストグラムの増減を調べることによってそれが公平かどうかがわかります。


`matplotlib.pyplot.hist()`に`cumulative=True`を指定すると累積ヒストグラムを作成することができます。

```Python
matplotlib.pyplot.hist(リスト型のデータ列, cumulative=True)
```

#### 例

- データ列の変数`data`を用いて、正規化されたビン数100の累積ヒストグラムを作成します。


```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
data = np.random.randn(10000)

# 正規化されたビン数100の累積ヒストグラムを作成
plt.hist(data, bins=100, density=True, cumulative=True)
plt.show()
```


![png](output_91_0.png)

## 散布図

### 散布図を作成する

<b style='color: #AA0000'>散布図</b>は`matplotlib.pyplot.scatter()`を用いて描画します。
横軸のデータ`x`とこれに対応する縦軸のデータ`y`を指定します。

```Python
matplotlib.pyplot.scatter(x, y)
```

#### 例

- リスト型の変数`x`、`y`のデータを平面上のx軸、y軸にそれぞれ対応させた散布図を作成


```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.choice(np.arange(100), 100)
y = np.random.choice(np.arange(100), 100)

# 散布図を作成
plt.scatter(x, y)

plt.show()
```


![png](output_102_0.png)

### マーカーの種類と色を設定する

横軸のデータ`x`、縦軸のデータ`y`に加え、`marker="指定子"`を指定すると<b>マーカーの種類（形）を設定でき</b>、`color="指定子"`を指定すると<b>マーカーの色を設定できます</b>。

```Python
matplotlib.pyplot.scatter(x, y, marker="マーカーの種類", color="マーカーの色")
```

以下は指定できるマーカーの種類とその色の一部です。

<b>マーカー</b>
- `"o"`: 円
- `"s"`: 四角
- `"p"`: 五角形
- `"*"`: 星
- `"+"`: プラス
- `"D"`: ダイアモンド

<b>色</b>
- `"b"` : 青
- `"g"` : 緑
- `"r"` : 赤
- `"c"` : シアン
- `"m"` : マゼンタ
- `"y"` : 黄色
- `"k"` : 黒
- `"w"` : 白

#### 例

- リスト型の変数`x`、`y`のデータを平面上のx軸、y軸にそれぞれ対応させた散布図を作成。
- マーカーの種類を四角、色を赤に設定してプロットしてください。


```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.choice(np.arange(100), 100)
y = np.random.choice(np.arange(100), 100)

# マーカーの種類を四角、色を赤に設定して散布図を作成
plt.scatter(x, y, marker="s", color="r")

plt.show()
```


![png](output_112_0.png)

### 値に応じてマーカーの大きさを設定する

横軸のデータ`x`、縦軸のデータ`y`に加え、`s=マーカーのサイズ`を指定すると<b>マーカーの大きさを設定できます</b>。デフォルト値は`20`です。
これを応用して、プロットデータに対応するリスト型のデータを`s`に指定します。すると指定した<b>リスト型のデータの値に応じてマーカーの大きさを個々に設定することができます</b>。
```Python
matplotlib.pyplot.scatter(x, y, s=マーカーのサイズ)
```

#### 例

- 変数`x`、`y`の値を散布図にプロットし、マーカーの大きさを変数`z`に応じた値に設定


```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.choice(np.arange(100), 100)
y = np.random.choice(np.arange(100), 100)
z = np.random.choice(np.arange(100), 100)

# マーカーの大きさを変数zに応じた値で個々に変わるようプロット
plt.scatter(x, y, s=z)

plt.show()
```


![png](output_122_0.png)


### 値に応じてマーカーの濃さを設定する

プロットデータに応じてマーカーの大きさを変えると、見難くなることがあります。その場合は、プロットデータに応じてマーカーの色の濃さを変えると効果的です。

横軸のデータ`x`、縦軸のデータ`y`に加え、`c=マーカーの色`を指定すると<b>マーカーの色を設定できます</b>。<br>
また、プロットデータに対応するリスト型のデータを`c`に指定し、さらに`cmap="色系統指定子"`を指定すると、<b>`c`の値に応じた濃さでマーカーをグラデーション表示することができます</b>。

```Python
matplotlib.pyplot.scatter(x, y, c=マーカーの色 または プロットデータに対応するリスト型のデータ, cmap="色系統指定子")
```
以下は使用できる色系統のうちの一部です。

<b>色系統指定子</b>
- "Reds": 赤
- "Blues": 青
- "Greens": 緑
- "Purples": 紫

#### 例

- 変数`x`、`y`の値を散布図にプロットし、変数`z`に応じた値で青系統の色でグラデーション表示してください。


```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.choice(np.arange(100), 100)
y = np.random.choice(np.arange(100), 100)
z = np.random.choice(np.arange(100), 100)

# 変数zに応じた値で、マーカーの色を青系統のグラデーションで表示してください
plt.scatter(x, y, c=z, cmap="Blues")

plt.show()
```


![png](output_132_0.png)

### カラーバーを表示する

プロットデータの大小に応じてマーカーを着色するだけでは、そのデータの水準や格差が分かりません。そこで、カラーバーを表示するとマーカーの濃さでだいたいの値が分かるようになります。

```Python
matplotlib.pyplot.colorbar()
```

#### 例

- 変数`x`、`y`の値を散布図にプロットし、変数`z`に応じた値で青系統の色でグラデーション表示し、カラーバーを表示します。


```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.choice(np.arange(100), 100)
y = np.random.choice(np.arange(100), 100)
z = np.random.choice(np.arange(100), 100)

# 変数zに応じた値で、マーカーの色を青系統のグラデーションで表示
plt.scatter(x, y, c=z, cmap="Blues")

# カラーバーを表示
plt.colorbar()

plt.show()
```


![png](output_142_0.png)


## 円グラフ

### 円グラフを作成する

<b style='color: #AA0000'>円グラフ</b>は`matplotlib.pyplot.pie()`を用いて描画します。グラフを円形にするには、`matplotlib.pyplot.axis("equal")`が必要です。<b>このコードがないと楕円になって</b>しまいます。

```Python
plt.pie(リスト型のデータ)
plt.axis("equal")
```

#### 例

- 変数`data`を円グラフで描画してください。

```python
import matplotlib.pyplot as plt

data = [60, 20, 10, 5, 3, 2]

# 変数dataを円グラフで描画
plt.pie(data)

# 円グラフを楕円から円形に変換
plt.axis("equal")

plt.show()
```


![png](output_153_0.png)

### 円グラフにラベルを設定する

`matplotlib.pyplot.pie()`に`labels=[ラベルのリスト]`を指定すると<b>ラベルを設定できます</b>。

```Python
matplotlib.pyplot.pie(データ, labels=[ラベルのリスト])
```

#### 例

- 変数`data`を円グラフで描画し、ラベルとして変数`labels`を設定します。

```python
import matplotlib.pyplot as plt

data = [60, 20, 10, 5, 3, 2]
labels = ["Apple", "Orange", "Banana", "Pineapple", "Kiwifruit", "Strawberry"]

# 変数dataを円グラフで描画 ラベルは変数labels
plt.pie(data, labels=labels)

plt.axis("equal")
plt.show()
```


![png](output_163_0.png)

### 特定の要素を目立たせる

円グラフの特徴的な要素だけを切り離して目立たせたい場合があります。
`matplotlib.pyplot.pie()`に`explode=[目立たせ度合いのリスト]`を指定すると<b>任意の要素を切り離して表示できます</b>。
「目立たせ度合い」には0から1の値をリスト型のデータで指定します。

```Python
matplotlib.pyplot.pie(データ, explode=[目立たせ度合いのリスト])
```

#### 例

- 変数`data`を円グラフで描画。
- ラベルとして変数`labels`を設定。
- 「目立たせ度合い」のリスト型のデータは変数`explode`です。


```python
import matplotlib.pyplot as plt

data = [60, 20, 10, 5, 3, 2]
labels = ["Apple", "Orange", "Banana", "Pineapple", "Kiwifruit", "Strawberry"]
explode = [0, 0, 0.1, 0, 0, 0]

# 変数dataに変数labelsのラベルを指定し、Bananaを目立たせた円グラフを描画
plt.pie(data, labels=labels, explode=explode)

plt.axis("equal")
plt.show()
```


![png](output_173_0.png)

## 3Dグラフ

### 3D Axesを作成する

ここでは<b>3Dグラフの描画</b>についてです。

3Dグラフを描画するには、<b>3D描画機能を持ったサブプロット</b>を作成する必要があり、サブプロットを作成する際に`projection="3d"`と指定します。

```Python
import matplotlib
matplotlib.pyplot.figure().add_subplot(1, 1, 1, projection="3d")
```

#### 例

- 用意された変数`fig`を用いて、3D描画機能を持ったサブプロット`ax`を追加します。追加する際、図は分割しないでください。

```python
import numpy as np
import matplotlib.pyplot as plt

# 3D描画を行うために必要なライブラリ
from mpl_toolkits.mplot3d import Axes3D

t = np.linspace(-2*np.pi, 2*np.pi)
X, Y = np.meshgrid(t, t)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Figureオブジェクトを作成
fig = plt.figure(figsize=(6, 6))

# 3D描画機能を持ったサブプロットaxを追加
ax = plt.figure().add_subplot(1, 1, 1, projection="3d")

# プロットして表示
ax.plot_surface(X, Y, Z)
plt.show()
```


    <Figure size 432x432 with 0 Axes>



![png](output_184_1.png)

### 曲面を作成する

できるだけ真に近い見た目のグラフを描画したい場合、`plot_surface()` にx軸、y軸、z軸に対応するデータを指定して<b>曲面を描画</b>します。

```Python
# サブプロットが変数`ax`の場合
ax.plot_surface(X, Y, Z)
```
描画したグラフは`matplotlib.pyplot.show()`を用いて画面に出力します。

#### 例

- 変数`X`,`Y`,`Z`のデータをそれぞれx軸、y軸、z軸に対応させて曲面をグラフに描画します。


```python
import numpy as np
import matplotlib.pyplot as plt

# 3D描画を行うために必要なライブラリ
from mpl_toolkits.mplot3d import Axes3D

x = y = np.linspace(-5, 5)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2)/2) / (2*np.pi)

# Figureオブジェクトを作成
fig = plt.figure(figsize=(6, 6))
# 3D描画機能を持ったサブプロットaxを追加
ax = fig.add_subplot(1, 1, 1, projection="3d")

# 曲面を描画して表示
ax.plot_surface(X, Y, Z)

plt.show()
```


![png](output_194_0.png)

### 3Dヒストグラムを作成する

<b>3次元のヒストグラムや棒グラフ</b>は2つの要素の関係性を見出すのに有効な手法で、データセットの各要素をそれぞれx軸とy軸に対応させ、z軸方向に積み上げて表現します。
`bar3d()`にx軸、y軸、z軸の位置と変化量に対応するデータを指定します。

```Python
# サブプロットが変数`ax`の場合
ax.bar3d(xpos, ypos, zpos, dx, dy, dz)
```

#### 例

3Dヒストグラムを作成します。
- x軸, y軸, z軸に対応する位置データはそれぞれ変数xpos, ypos, zpos です。
- 増加量は変数dx, dy, dz です。

```python
import matplotlib.pyplot as plt
import numpy as np

# 3D描画を行うために必要なライブラリ
from mpl_toolkits.mplot3d import Axes3D

# Figureオブジェクトを作成
fig = plt.figure(figsize=(5, 5))
# サブプロットaxを追加
ax = fig.add_subplot(111, projection="3d")

# x, y, zの位置を決める
xpos = [i for i in range(10)]
ypos = [i for i in range(10)]
zpos = np.zeros(10)

# x, y, zの変化量を決める
dx = np.ones(10)
dy = np.ones(10)
dz = [i for i in range(10)]

# 3次元のbarを作成
ax.bar3d(xpos, ypos, zpos, dx, dy, dz)

plt.show()
```


![png](output_204_0.png)

### 3D散布図を作成する

<b>3次元の散布図</b>は互いに関係を持っている（または持っていると思われる）3種類のデータを3次元の空間上にプロットすることで<b>データの傾向を視覚的に予測するのに有効</b>です。
`scatter3D()`にx軸、y軸、z軸に対応するデータを指定します。ただし、指定するデータは1次元でなければならないため、1次元のデータではない場合あらかじめ`np.ravel()`を用いてデータを変換します。

```Python
x = np.ravel(X)
# サブプロットが変数`ax`の場合
ax.scatter3D(x, y, z)
```

#### 例

変数`X`, `Y`, `Z`ついてあらかじめ`np.ravel()`を用いて、それぞれ変数`x`, `y`, `z`に一次元に変換したデータをおいています。
- 3D散布図を作成してください。x軸, y軸, z軸に対応するデータはそれぞれ変数`x`, `y`, `z`です。


```python
import numpy as np
import matplotlib.pyplot as plt

# 3D描画を行うために必要なライブラリ
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)
X = np.random.randn(1000)
Y = np.random.randn(1000)
Z = np.random.randn(1000)

# Figureオブジェクトを作成
fig = plt.figure(figsize=(6, 6))
# サブプロットaxを追加
ax = fig.add_subplot(111, projection="3d")

# 3D散布図を作成
ax.scatter3D(X, Y, Z)

plt.show()
```


![png](output_214_0.png)

### 3Dグラフにカラーマップを適用する

色が単調な3Dグラフは凹凸が多い部分など見にくい場合があります。その場合は<b>グラフの点がとる座標に応じて表示する色を変える機能</b>を使用して見やすくすることができます。
あらかじめ`matplotlib`から`cm`を`import`しておきます。データをプロットする際、`plot_surface()`に`cmap=cm.coolwarm`を指定すると、<b>z軸の値にカラーマップを適用できます</b>。

```Python
import matplotlib.cm as cm
# サブプロットが変数`ax`の場合
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
```

#### 例

変数`X`, `Y`, `Z` にはそれぞれx軸、y軸、z軸に対応したデータが用意されています。
- サブプロットax にX, Y, Z をプロットし、z軸の値にカラーマップを適用してください。


```python
import numpy as np
import matplotlib.pyplot as plt

# 3D描画を行うために必要なライブラリ
from mpl_toolkits.mplot3d import Axes3D
# カラーマップを表示するためのライブラリ
from matplotlib import cm

t = np.linspace(-2*np.pi, 2*np.pi)
X, Y = np.meshgrid(t, t)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Figureオブジェクトを作成
fig = plt.figure(figsize=(6, 6))
# サブプロットaxを追加
ax = fig.add_subplot(111, projection="3d")

# サブプロットaxにzの値にカラーマップを適用
ax.plot_surface(X, Y, Z, cmap="coolwarm")

plt.show()
```

![png](output_224_0.png)
