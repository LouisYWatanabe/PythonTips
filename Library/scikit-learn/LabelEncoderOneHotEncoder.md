---
title: Pythonでのカテゴリ変数(名義尺度・順序尺度)のエンコード(数値化)方法 ～順序のマッピング、LabelEncoderとOne Hot Encoder～
tags: Python 前処理 scikit-learn pandas
author: uratatsu
slide: false
---
#カテゴリデータの前処理
性別や血液型、郵便番号などの名義尺度、サイズや階層、評価など順序に意味のある順序尺度はカテゴリデータと呼ばれる。データ分析や機械学習アルゴリズムに使用するためには、数値化しなければならないので、そこらへんのやり方をまとめておく。


#順序尺度の数値化
ファーストフード店の飲み物のサイズ　「L>M>S」

などは順序に意味があるため、順序尺度となる。カテゴリデータのなかでも、順序尺度については
S ⇒　1
M ⇒　2
L ⇒　3
として、わかりやすく数値化できる。
順序尺度については、マッピングを用いて数値化する。

このような飲み物リストがあったとして

```python:Categorical_Encoding.py
import pandas as pd
df = pd.DataFrame([
        ['S', 100, 'cola'],
        ['M', 150, 'tea'],
        ['L', 200, 'coffee']])
df.columns = ['size', 'price', 'label']
df
```
| size       |price     |label    |
|:-----------------|:-------------|:------|
| S             | 100 |        cola        |
| M           | 150 |       tea       |
| L            |  200|        coffee        |

飲み物サイズのマッピングを行う。

```python:Categorical_Encoding.py
size_mapping = {'L': 3, 'M': 2, 'S': 1}
df['size'] = df['size'].map(size_mapping)
df
```
| size       |price     |label    |
|:-----------------|:-------------|:------|
| 1             | 100 |        cola        |
| 2           | 150 |       tea       |
| 3            |  200|        coffee        |

順序尺度については、ラベルと整数を対応させるディクショナリを使って、マッピングを行うことで数値化する。

#名義尺度の数値化
性別や血液型などの名義尺度については、順序尺度のように順序がないので、サイズのように、どのサイズがどの整数になるのか対応させる必要はない。先ほどと同様にディクショナリを生成してマッピングしてもよいが、scikit-learnにあるLabelEncoderというクラスを使えば、勝手に変換してくれる。

#LabelEncodrの使い方
LabelEncoderは、ラベルを0～クラスの種類数n-1の値に変換してくれる。
array型なら何でもいけるので数値を変換することもできるが、文字列を数値に変換したいときに使うのが主な使い方だろう。

## $Example$
```python
from sklearn.preprocessing import LabelEncoder

# 変数Sex, Embarkedにlabel encodingを適用する
for c in ['Sex', 'Embarked']:
    le = LabelEncoder()
    # ラベルを覚えさせる
    le = le.fit(train_X[c].fillna('NA'))
    # 変換
    train_X[c] = le.transform(train_X[c].fillna('NA'))
    test_X[c] = le.transform(test_X[c].fillna('NA'))
# 表示確認
for c in ['Sex', 'Embarked']:
    print(train_X[c].head())
```
```
0    1
1    0
2    0
3    0
4    1
Name: Sex, dtype: int64
0    3
1    0
2    3
3    3
4    3
Name: Embarked, dtype: int64
```


```python
from sklearn.preprocessing import LabelEncoder
#LabelEncoderのインスタンスを生成
le = LabelEncoder()
#ラベルを覚えさせる
le = le.fit(df['label'])
#ラベルを整数に変換
df['label'] = le.transform(df['label'])
df
```
| size       |price     |label    |
|:-----------------|:-------------|:------|
| 1             | 100 |        1        |
| 2           | 150 |       2       |
| 3            |  200|        0        |

cola ⇒1
tea ⇒2
coffee ⇒0
に変換された。数字の大小に意味はなく、coffeeとteaの平均がcolaになったりはしないし、coffeeのほうがcolaより小さいわけでもない。
順序に意味がない名義尺度では、LabelEncoderで変換される整数値を用いて回帰分析を行うことはできない。上の例で、label と sizeを使ってpriceを回帰分析で予測する場合、labelは1⇒2⇒0の順で大きくなるので、変数としてふさわしくない。LabelEncoderで1つの変数の離散値として扱う場合は、決定木分析、ランダムフォレストなどの分析と相性が良い。また、後述するダミー変数よりも変数の数を節約できる。

#One Hot Encoderの使い方
LabelEncoderでは、回帰分析に適用できないため、名義尺度の数値化は、One Hot Encoderを用いてダミー変数化する。label列のcola,tea,coffeeを3つの新しい変数に変換し、二値の組み合わせによって、飲み物の種類を示すようにする。例えばcolaは、cola=1,tea=0,coffee=0としてエンコードできる。

```python
from sklearn.preprocessing import OneHotEncoder
#one-hot エンコーダの生成
oe = OneHotEncoder(categorical_features=[2])
#one-hot エンコーディングを実行
array = oe.fit_transform(df).toarray()
array
```
    array([[  0.,   1.,   0.,   1., 100.],
        [  0.,   0.,   1.,   2., 150.],
        [  1.,   0.,   0.,   3., 200.]])

One Hot Encoder は整数値しかダミー変数に変換してくれないので、文字の変換の場合は、先ほどのLabelEncoderと合わせて使用する。
categorical_features=[2]で3番目の列をダミー変数に変換する指定をしている。.toarray()でarray型に成型し、以下でDataFrame型に変換する。

```python:Categorical_Encoding.py
#DataFrameに変換
columns = ['label_coffee','label_cola','label_tea','size','price']
df1 = pd.DataFrame(data = array, columns = columns)
#列入れ替え
df1 = df1[['size','price','label_cola','label_tea','label_coffee']]
df1
```
One Hot Encoderでのダミー変数化は以上の通りだが、One Hot Encoderでは、一度LabelEncoderを通さなきゃいけなかったり、順番を成型したりいろいろめんどくさいので、pandasのget_dummiesメソッドを使うことが多い。

#get_dummiesの使い方
get_dummiesは、文字列のまま使用できるので、いったんlabel列を文字列に戻す。

```python:Categorical_Encoding.py
df['label'] = le.inverse_transform(df['label'])
df
```
| size       |price     |label    |
|:-----------------|:-------------|:------|
| 1             | 100 |        cola        |
| 2           | 150 |       tea       |
| 3            |  200|        coffee        |

get_dummiesでlabel列をダミー変数化して、元のDataFrameに結合する。

```python:Categorical_Encoding.py
df_dummy = pd.get_dummies(df['label'])
df2 = pd.concat([df.drop(['label'],axis=1),df_dummy],axis=1)
df2
```
| size       |price     |coffee    |cola     |tea|
|:-----------------|:-------------|:------|:------|:------|
| 1             | 100 |        0        |1|0|
| 2           | 150 |      0       |0|1|
| 3            |  200|        1        |0|0|
これでダミー変数化は完成した。

#多重共線性
最後に注意したいのが、回帰分析などにダミー変数を使用したい場合は、変数同士の相関が高いと多重共線性という問題が発生するため、one-hotエンコーディングの配列から列の1つを削除して使用しなければならない。例えばCoffee列を削除しても、cola列とtea列がともに0であればcoffeeであることがわかるので、情報としての欠落はない。
get_dummieのdrop_firstパラメータにTrueを渡すと最初の列を削除できる。

```python:Categorical_Encoding.py
df_dummy = pd.get_dummies(df['label'],drop_first = True)
df3 = pd.concat([df.drop(['label'],axis=1),df_dummy],axis=1)
df3
```
| size       |price     |cola     |tea|
|:-----------------|:------|:------|:------|
| 1             | 100 |1|0|
| 2           | 150 |0|1|
| 3            |  200|0|0|

以上、pythonでのカテゴリ変数(名義尺度・順序尺度)のエンコード方法でした。
