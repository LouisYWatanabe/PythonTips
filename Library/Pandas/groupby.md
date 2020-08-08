---
title: Pandas の groupby の使い方
tags: pandas GroupBy Python Jupyter
author: propella
slide: false
---
Python でデータ処理するライブラリの定番 Pandas の groupby がなかなか難しいので整理する。特に apply の仕様はパラメータの関数の戻り値によって予想外の振る舞いをするので凶悪に思える。

まず必要なライブラリを import する。


```python
import pandas as pd
import numpy as np
```

## DataFrame を作る

サンプル用のデータを適当に作る。


```python
df = pd.DataFrame({
    'city': ['osaka', 'osaka', 'osaka', 'osaka', 'tokyo', 'tokyo', 'tokyo'],
    'food': ['apple', 'orange', 'banana', 'banana', 'apple', 'apple', 'banana'],
    'price': [100, 200, 250, 300, 150, 200, 400],
    'quantity': [1, 2, 3, 4, 5, 6, 7]
})
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>food</th>
      <th>price</th>
      <th>quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>osaka</td>
      <td>apple</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>osaka</td>
      <td>orange</td>
      <td>200</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>osaka</td>
      <td>banana</td>
      <td>250</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>osaka</td>
      <td>banana</td>
      <td>300</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tokyo</td>
      <td>apple</td>
      <td>150</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tokyo</td>
      <td>apple</td>
      <td>200</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>tokyo</td>
      <td>banana</td>
      <td>400</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



余談だが、本題に入る前に Pandas の二次元データ構造 DataFrame について軽く触れる。余談だが Pandas は列志向のデータ構造なので、データの作成は縦にカラムごとに行う。列ごとの処理は得意で速いが、行ごとの処理はイテレータ等を使って Python の世界で行うので遅くなる。

DataFrame には index と呼ばれる特殊なリストがある。上の例では、`'city', 'food', 'price'` のように各列を表す index と `0, 1, 2, 3, ...` のように各行を表す index がある。また、各 index の要素を label と呼ぶ。それぞれ以下のようなプロパティで取得出来る。


```python
df.columns # 列 label の取得
```




    Index(['city', 'food', 'price', 'quantity'], dtype='object')




```python
df.index # 行 label の取得
```




    RangeIndex(start=0, stop=7, step=1)



## 簡単な groupby の使い方

余談終わり。groupby は、同じ値を持つデータをまとめて、それぞれの塊に対して共通の操作を行いたい時に使う。例えば一番簡単な使い方として、city ごとの price の平均を求めるには次のようにする。groupby で出来た [GroupBy](https://pandas.pydata.org/pandas-docs/stable/api.html#groupby) オブジェクトに対して、平均をとる [mean](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.core.groupby.GroupBy.mean.html#pandas.core.groupby.GroupBy.mean) メソッドを呼ぶと良い。


```python
df.groupby('city').mean()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>quantity</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>osaka</th>
      <td>212.5</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>tokyo</th>
      <td>250.0</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>



グループの指定に複数の label を指定する事も出来る。city と food の組み合わせで平均をとるには次のようにする。


```python
df.groupby(['city', 'food']).mean()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>price</th>
      <th>quantity</th>
    </tr>
    <tr>
      <th>city</th>
      <th>food</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">osaka</th>
      <th>apple</th>
      <td>100.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>banana</th>
      <td>275.0</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>orange</th>
      <td>200.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">tokyo</th>
      <th>apple</th>
      <td>175.0</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>banana</th>
      <td>400.0</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>



groupby を使うと、デフォルトでグループラベルが index になる。index にしたく無い場合は `as_index=False` を指定する。


```python
df.groupby(['city', 'food'], as_index=False).mean()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>food</th>
      <th>price</th>
      <th>quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>osaka</td>
      <td>apple</td>
      <td>100.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>osaka</td>
      <td>banana</td>
      <td>275.0</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>osaka</td>
      <td>orange</td>
      <td>200.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tokyo</td>
      <td>apple</td>
      <td>175.0</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tokyo</td>
      <td>banana</td>
      <td>400.0</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>



## GroupBy オブジェクトの性質

デバッグ以外で使うところは無いかも知れないが、groupby によって作られた GroupBy オブジェクトの性質を調べるプロパティが幾つかある。まず、groupby によってどのように DataFrame が分割されたかを知るには groups を使う。`{ 列 label: [行 label, 行 label, ...], ... }` のような形で、どのグループにどの列が入ったか分かる。


```python
df.groupby('city').groups
```




    {'osaka': Int64Index([0, 1, 2, 3], dtype='int64'),
     'tokyo': Int64Index([4, 5, 6], dtype='int64')}



あるグループにどのようなデータが入ったかを知るには get_group を使う。


```python
df.groupby('city').get_group('osaka')
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>food</th>
      <th>price</th>
      <th>quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>osaka</td>
      <td>apple</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>osaka</td>
      <td>orange</td>
      <td>200</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>osaka</td>
      <td>banana</td>
      <td>250</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>osaka</td>
      <td>banana</td>
      <td>300</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



各グループのサイズは size で取得出来る。


```python
df.groupby('city').size()
```




    city
    osaka    4
    tokyo    3
    dtype: int64



size の結果は [Series](https://pandas.pydata.org/pandas-docs/stable/api.html#series) という一次元列を表すオブジェクトが返る。Series を使うと、osaka グループのサイズは添字を使って取得出来る。


```python
df.groupby('city').size()['osaka']
```




    4



## さまざまな Aggregation

GroupBy.mean() のように、グループごとに値を求めて表を作るような操作を Aggregation と呼ぶ。このように GroupBy オブジェクトには Aggregation に使う関数が幾つか定義されているが、これらは [agg()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.core.groupby.DataFrameGroupBy.agg.html#pandas.core.groupby.DataFrameGroupBy.agg) を使っても実装出来る。


```python
df.groupby('city').agg(np.mean)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>quantity</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>osaka</th>
      <td>212.5</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>tokyo</th>
      <td>250.0</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>



agg には多様な使い方がある。上の例では、mean() を使って各グループごとに price と quantity 両方の平均を求めたが、例えば price の平均と quantity の合計を同時に知りたいときは以下のように { グループ名: 関数 } の dict を渡す。関数には Series を受け取って一つの値を返す物を期待されている。 


```python
def my_mean(s):
    """わざとらしいサンプル"""
    return np.mean(s)

df.groupby('city').agg({'price': my_mean, 'quantity': np.sum})
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>quantity</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>osaka</th>
      <td>212.5</td>
      <td>10</td>
    </tr>
    <tr>
      <th>tokyo</th>
      <td>250.0</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>



## Group ごとに複数行を返す

Aggregation の結果はグループごとに一行にまとめられるが、もっと柔軟に結果を作りたいときは [apply](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.core.groupby.GroupBy.apply.html#pandas.core.groupby.GroupBy.apply) を使う。apply に渡す関数には get_group で得られるようなグループごとの DataFrame が渡される。グループ名は df.name で取得出来る。

### apply 関数の結果としてスカラを返す場合。全体の結果は Series になる。

* groupby で作った label が結果の row index になる。
* 行数はグループの数と同じになる。
* as_index の効果は無い。


```python
df.groupby(['city', 'food'], as_index=False).apply(lambda d: (d.price * d.quantity).sum())
```




    city   food  
    osaka  apple      100
           banana    1950
           orange     400
    tokyo  apple     1950
           banana    2800
    dtype: int64



グループ名 にアクセスしてみた例


```python
df.groupby(['city', 'food'], as_index=False).apply(lambda d: d.name)
```




    city   food  
    osaka  apple      (osaka, apple)
           banana    (osaka, banana)
           orange    (osaka, orange)
    tokyo  apple      (tokyo, apple)
           banana    (tokyo, banana)
    dtype: object



### apply 関数の結果として Series を返す場合。全体の結果は Series になる。

* groupby で作った label に加えて、apply 関数の結果の index が結果全体の row index になる。
* 全体の行数は関数から返す結果に依存する。
* as_index=False を指定すると、index が消えて連番になる。


```python
def total_series(d):
    return d.price * d.quantity

df.groupby(['city', 'food']).apply(total_series)
```




    city   food     
    osaka  apple   0     100
           banana  2     750
                   3    1200
           orange  1     400
    tokyo  apple   4     750
                   5    1200
           banana  6    2800
    dtype: int64



### apply 関数の結果として元の row index を保存した DataFrame を返す場合

DataFrame を返す場合、返す DataFrame に含まれる row index によって振る舞いが違う。非常に凶悪な仕様!!!!

元の index を保存した場合、下記 Transformation と同じ動作ように groupby の label は消える。

* apply 関数の結果を連結した DataFrame が作られる。groupby で対象になる label は index にならない。
* as_index=False の効果なし


```python
def total_keepindex(d):
    return pd.DataFrame({
        'total': d.price * d.quantity # ここで返る DataFrame の row index は d の row index と同じ
    })

df.groupby(['city', 'food']).apply(total_keepindex)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>400</td>
    </tr>
    <tr>
      <th>2</th>
      <td>750</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>750</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1200</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2800</td>
    </tr>
  </tbody>
</table>
</div>



### apply 関数の結果として元の row index を保存しない DataFrame を返す場合

元の index を保存しないと groupby で作った label が結果の row index になる。

* groupby で作った label が結果の row index になる。
* as_index=False の効果あり


```python
def total_keepnoindex(d):
    return pd.DataFrame({
        'total': (d.price * d.quantity).sum()
    }, index=['hoge'])
df.groupby(['city', 'food']).apply(total_keepnoindex)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>total</th>
    </tr>
    <tr>
      <th>city</th>
      <th>food</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">osaka</th>
      <th>apple</th>
      <th>hoge</th>
      <td>100</td>
    </tr>
    <tr>
      <th>banana</th>
      <th>hoge</th>
      <td>1950</td>
    </tr>
    <tr>
      <th>orange</th>
      <th>hoge</th>
      <td>400</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">tokyo</th>
      <th>apple</th>
      <th>hoge</th>
      <td>1950</td>
    </tr>
    <tr>
      <th>banana</th>
      <th>hoge</th>
      <td>2800</td>
    </tr>
  </tbody>
</table>
</div>



### 注意!! apply 関数が一度も呼ばれないとカラムが出来ない

Pandas の凶悪な所でありまた動的型付け言語の欠点なのだが、apply 関数の結果で動的にカラムを決めているからか、ゼロ行の DataFrame に対して apply を実行するとカラムが作成されない。ゼロ行だけ特別扱いしないと行けないので分かりづらいバグを生む。

例えばこの場合3行の DataFrame の場合カラムが出来る。


```python
pd.DataFrame({'hoge': [1,1,3], 'fuga': [10, 20, 30]}).groupby('hoge').apply(np.sum)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hoge</th>
      <th>fuga</th>
    </tr>
    <tr>
      <th>hoge</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>



ところがゼロ行の DataFrame に対して同じ apply を実行するとカラムが消えてしまう。


```python
pd.DataFrame({'hoge': [], 'fuga': []}).groupby('hoge').apply(np.sum)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
    </tr>
    <tr>
      <th>hoge</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



## Transformation

グループごとの統計情報を使ってすべての行を集計したい場合は Transformation を使う。説明が難しい。。。transformation の引数にはグループごとの列の Series が与えられる。戻り値は引数と同様の Series かスカラを渡す。スカラを渡した場合は引数と同じ個数だけ繰り返される。

例えば、グループごとに各アイテムの割合を求めるには次のようにする。


```python
def transformation_sample(s):
    return (s / s.sum() * 100).astype(str) + '%'

df.groupby(['city']).transform(transformation_sample)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11.76470588235294%</td>
      <td>10.0%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>23.52941176470588%</td>
      <td>20.0%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29.411764705882355%</td>
      <td>30.0%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.294117647058826%</td>
      <td>40.0%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.0%</td>
      <td>27.77777777777778%</td>
    </tr>
    <tr>
      <th>5</th>
      <td>26.666666666666668%</td>
      <td>33.33333333333333%</td>
    </tr>
    <tr>
      <th>6</th>
      <td>53.333333333333336%</td>
      <td>38.88888888888889%</td>
    </tr>
  </tbody>
</table>
</div>



## 参考

* ガイド: [Group By: split-apply-combine](https://pandas.pydata.org/pandas-docs/stable/groupby.html)
* groupby 関数のリファレンス: [pandas.DataFrame.groupby](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html)
* GroupBy オブジェクトのリファレンス: [GroupBy](https://pandas.pydata.org/pandas-docs/stable/api.html#groupby)
* [Series](https://pandas.pydata.org/pandas-docs/stable/api.html#series)
* [DataFrame](https://pandas.pydata.org/pandas-docs/stable/api.html#dataframe)

