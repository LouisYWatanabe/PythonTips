

```python
%matplotlib inline
```

# 目次

- **[1.問題設定](#1.問題設定)**
- **[2.データ](#2.データ)**
- **[3.コード](#3.コード)**
- **[4.結果](#4.結果)**

# 1.問題設定

### アヤメの「花びらの長さ」,「花びらの幅」の情報から品種を予測したい

# 2.データ

## 以下のアヤメのデータを使用する

#### データ説明（アヤメのデータセット）
---

Irisデータには、150個のアヤメ(花の一種)のサンプルの「がく片の長さ」「がく片の幅」「花びらの長さ」「花びらの幅」の４つの要素(単位はcm)と、3種の品種が格納されています。


| ID 	| がく片の長さ 	| がく片の幅 	| 花びらの長さ 	| 花びらの幅 	| 品種クラス    	|
|----	|----------	|----------	|--------------	|------------	|---------------	|
| 0  	| 5.1      	| 3.5      	| 1.4          	| 0.2        	| Iris-setosa     	|
| 1  	| ・       	| ・       	| ・           	| ・         	| Iris-versicolor	|
|  ・	| ・       	| ・       	| ・           	| ・         	| Iris-setosa     	|
| ・  	| ・       	| ・       	| ・           	| ・         	| Iris-virginica	|

Irisデータのイメージ図（実際の値とは違います）


# 3.コード


```python
# データの読み込み
import pandas as pd
'''
# ネットワークを使用してデータを取得する場合（中身はpd.read_csvのデータと同じです）
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
'''
df = pd.read_csv("../data/iris.data", header=None)

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



## 前処理


```python
# 欠損値の有無の確認
df.isnull().sum()
```




    0    0
    1    0
    2    0
    3    0
    4    0
    dtype: int64




```python
# データの型の確認
df.dtypes
```




    0    float64
    1    float64
    2    float64
    3    float64
    4     object
    dtype: object



データの4行目（品種）（目的変数）がobject型で、<br>このままでは決定木での学習ができないので数値変換します


```python
import numpy as np
```


```python
# 品種ラベルを数値変換してplotする

# from sklearn.datasets import load_iris
# iris = load_iris()
# iris.target

import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
# plt.xkcd()

'''
前処理
'''
# データの4行目（品種）を目的変数として取得し .valueでarrayに変換
y = df.iloc[:, 4].values

# 目的変数の品種「Setosa」を0、「Versicolor」を1「Virginica」を2に変換
# y[y == "Iris-setosa"] = 0
# y[y == "Iris-versicolor"] = 1
# y[y == "Iris-virginica"] = 2

# unique()でy要素の個数ごとにenumerate()でindex付きデータとして取得
# 目的変数の品種「Setosa」を0、「Versicolor」を1「Virginica」を2に変換
for idx, cl in enumerate(np.unique(y)):
    y[y == cl] = idx

# 3行目（花びらの長さ）と4行目（花びらの幅）を説明変数として取得し .valueでarrayに変換
X = df.iloc[:, [2,3]].values

'''
可視化
'''
# setosaのプロット(赤の○)
plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="setosa")

# versicolorのプロット(青の✕)
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker="x", label="versicolor")
# virginicaのプロット(緑の□)
plt.scatter(X[101:150,0],X[101:150,1],color='green',marker='s',label='virginica')

# 軸のラベルの設定
# 花びらの長さ
plt.xlabel("petal length [cm]")
# 花びらの幅
plt.ylabel("petal width [cm]")
# 凡例の設定(左上に配置)
plt.legend(loc="upper left")
# グラフの位置サイズの自動調整
plt.tight_layout()
# 図の表示
plt.show()
```


![png](output_12_0.png)



```python
# 説明変数の種類の確認
# 前処理結果の確認
print("Class labels:", np.unique(y))
```

    Class labels: [0 1 2]
    


```python
# ndarray型なので.dtypeでデータ型を確認する
y.dtype
```




    dtype('O')




```python
# 目的変数の型変換（object -> int32）
y = y.astype(np.int32)
y.dtype
```




    dtype('int32')



## 学習


```python
'''
学習
'''
# 訓練データとテストデータに分割する
from sklearn.model_selection import train_test_split
# 全体の20%をテストデータとして分割
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y        # 層化抽出:均等に分割させたいデータの指定
)
```


```python
# 分割後のデータ数の確認 bincount()で出現件数を取得
print("Labels counts in y:", np.bincount(y))
print("Labels counts in y_train:", np.bincount(y_train))
print("Labels counts in y_test:", np.bincount(y_test))
```

    Labels counts in y: [50 50 50]
    Labels counts in y_train: [40 40 40]
    Labels counts in y_test: [10 10 10]
    


```python
from sklearn import tree
# 分類器モデルの構築
model = tree.DecisionTreeClassifier(criterion="gini", random_state=42)
# 学習の実施
tree_model = model.fit(X_train,y_train)
```

## 学習結果の可視化


```python
'''
可視化
ターミナルで「pip install pydotplus」を実行して、
graph_from_dot_dataを使えるようにする必要があります。
'''
try:
    from pydotplus import graph_from_dot_data
except ImportError as ex:
    print("Error: the pydotplus library is not installed.")
    # import pydotplusできなければ、pydotplusをインストールする
    !pip install pydotplus
    
    from pydotplus import graph_from_dot_data

dot_data = tree.export_graphviz(
    tree_model, 
    feature_names=["petal_length [cm]", "petal_width [cm]"],
    class_names=["Setosa", "Versicolor", "Virginica"],
    filled=True, # Trueにすると、分岐の際にどちらのノードに多く分類されたのか色で示してくれる
    rounded=True, # Trueにすると、ノードの角を丸く描画する。
    impurity=True, # Trueにすると、不純度を表示する。
    out_file=None # .dotファイルを作成する場合は'ファイル名.dot'でファイルを保存する
)

graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')
```




    True




```python
from IPython.display import Image
# 保存したpngファイルを表示
Image(filename='./tree.png', width=1000)
```




![png](output_22_0.png)




```python
'''
決定境界の可視化関数
'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    '''
    パラメータ
    X : shape = [n_samples, n_features]
        訓練データ、説明変数
    y : shape = [n_samples]
        目的変数
    test_idx : shape = [n_samples, n_features]
        テストデータの指定があるとき、他よりも目立たせて表示
    resolution : float
        表示解像度
    戻り値 : なし
    '''
    # マーカーとカラーマップの準備
    markers = ("o", "x", "^", "s", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])        # （分類した説明変数の数だけ）カラーマップの作成.unique()で出現頻度を取得

    # 決定領域の取得
    # 特徴量の最大値と最小値を取得
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # グリッド配列を作成（参考：https://deepage.net/features/numpy-meshgrid.html）
    # 訓練データと同じ個数のカラムを持つ行列を作成する
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # モデルの予測を実行する。そのままで予測ができない（モデルが2次元の特徴量で学習している）ため各特徴量を.ravel()で一次元に変換する
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # 予測結果をグラフ表示のために元のデータサイズに再変換
    Z = Z.reshape(xx1.shape)

    # meshgrid で作ったxx1とxx2、そして高さZを等高線contourに渡す
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    # 軸の範囲設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # クラスごとにサンプルを表示（.unique()で出現頻度を取得し、enumerate()でそのインデックス番号と要素を取得）
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=cmap(idx),
            marker=markers[idx],
            label=cl,
            edgecolors="black"
        )

    # もしテストデータの指定があればを他より目立たせる(点を○表示)
    if test_idx:
        # すべてのサンプルをプロット
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(
            X_test[:, 0], X_test[:, 1],
            c='',
            edgecolor='black',
            alpha=1.0,
            linewidth=1,
            marker='o',
            s=100,
            label='test set'
        )
```


```python
# 可視化用関数は上のセルで定義しているので以下の宣言は必要ない
# # 上の階層のcommonディレクトリにあるclassifier_plot.pyファイルをインポートする
# import sys
# # ファイルを検索する対象に上の階層も追加する
# sys.path.append("../")
# # 分類描画関数plot_decision_regions()のインポート
# from common.classifier_plot import plot_decision_regions

# 訓練データとテストデータの特徴量を行方向に結合
X_combined = np.vstack((X_train, X_test))
# 訓練データとテストデータのクラスラベルを結合
# 水平方向に(horizontal)連結
y_combined = np.hstack((y_train, y_test))

# 訓練データとテストデータ結合後の変数の中からtestデータの開始地点と終了地点を取得
test_str_pt = np.bincount(y).sum() - np.bincount(y_test).sum()
test_fin_pt = np.bincount(y).sum()

# 決定境界のプロット
plot_decision_regions(X=X_combined, y=y_combined, classifier=tree_model, test_idx=range(test_str_pt, test_fin_pt))

# 軸のラベルの設定
# 花びらの長さ
plt.xlabel("petal length [cm]")
# 花びらの幅
plt.ylabel("petal width [cm]")

# 凡例の設定(左上に配置)
plt.legend(loc="upper left")

# グラフを表示
plt.tight_layout()
plt.show()
```


![png](output_24_0.png)



```python
# 各特徴量の重要度を可視化
# (どの特徴量がどれくらい分割に寄与したのかを確認する)
n_features = X.shape[1] # 全説明変数

plt.barh(range(n_features), tree_model.feature_importances_, align="center") # 描画する際の枠組みを設定
plt.yticks(np.arange(n_features), ["petal_length [cm]", "petal_width [cm]"]) # 縦軸の設定
plt.show()

print("petal_length [cm]:", tree_model.feature_importances_[0])
print("petal_width  [cm]:", tree_model.feature_importances_[1])
```


![png](output_25_0.png)


    petal_length [cm]: 0.578297008343644
    petal_width  [cm]: 0.4217029916563561
    

### 交差検証

学習したモデルを使用し、交差検証して正解率を確認する


```python
from sklearn.model_selection import KFold, cross_val_score

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

tree_score = cross_val_score(tree_model, X, y, cv=kfold)

print(tree_score)
print(tree_score.mean())
```

    [1.         0.96666667 0.93333333 0.9        0.93333333]
    0.9466666666666667
    

## 予測


```python
'''
予測
'''
print("決定木の正解率 : {:.2f}".format(tree_model.score(X_test, y_test)))
```

    決定木の正解率 : 0.93
    


```python
# テストデータで予測を行う
y_pred = tree_model.predict(X_test)

print("誤分類したデータ数: %d"% (y_test != y_pred).sum())
print("予測したデータ数: %d"% (y_pred.sum()+1))
print("誤分類率: %.3f"% (((y_test != y_pred).sum()) / (y_pred.sum()+1)) )
# 正解率の表示
print("Accuracy: %.3f"% (1-((y_test != y_pred).sum()) / (y_pred.sum()+1)) )
```

    誤分類したデータ数: 2
    予測したデータ数: 31
    誤分類率: 0.065
    Accuracy: 0.935
    


```python
'''
正解率の表示
'''
print(__doc__)
from sklearn.metrics import accuracy_score

print("決定木の正解率 : {:.2f}".format(accuracy_score(y_test, y_pred)))
```

    
    正解率の表示
    
    決定木の正解率 : 0.93
    


```python
'''
適合率（precision）
'''
print(__doc__)
from sklearn.metrics import precision_score
# averageにデフォルトで2値分類用の'binary'が指定されているので、ここでは他の引数を設定します。
# マイクロ平均（micro）かマクロ平均（macro）
print("決定木の適合率 : {:.2f}".format(precision_score(y_test, y_pred, average="micro")))
```

    
    適合率（precision）
    
    決定木の適合率 : 0.93
    


```python
'''
再現率（recall）
'''
print(__doc__)
from sklearn.metrics import recall_score
# averageにマイクロ平均（micro）を設定
print("決定木の再現率 : {:.2f}".format(recall_score(y_test, y_pred, average="micro")))
```

    
    再現率（recall）
    
    決定木の再現率 : 0.93
    


```python
'''
F値（F1-measure）
'''
print(__doc__)
from sklearn.metrics import f1_score
# averageにマイクロ平均（micro）を設定
print("決定木のF値 : {:.2f}".format(f1_score(y_test, y_pred, average="micro")))
```

    
    F値（F1-measure）
    
    決定木のF値 : 0.93
    

# 4.結果

### 決定木を使用して、アヤメの花びらの情報から<br>93%の正解率を持つモデルを作ることができた。

<br>
<br>

## おまけ（チューニングして分類のされ方は変わったものの混同行列に変化がなかったので）

グリッドサーチを実施し、その中のベストパラメータを使用してモデルを作る


```python
from sklearn import tree
from sklearn.model_selection import GridSearchCV
 
# チューニングするパラメータ
tuned_parameters = {
    "criterion": ["gini", "entropy"],
    "max_depth":  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], # 木の深さを1-10でグリッドサーチ
    "max_leaf_nodes":  [2,4,6,8,10] # 最大終端ノード数を2,4,6,8,10でグリッドサーチ
}
 
# 上記で用意したパラメーターごとに交差検証を実施。最適な木の深さを確認する。
clf = GridSearchCV(
    tree.DecisionTreeClassifier(
        random_state=42,
        splitter="best"
    ),
    tuned_parameters, scoring="accuracy",
    cv=5, 
    n_jobs=-1
)

clf = clf.fit(X_train, y_train) # モデル作成
best_clf = clf.best_estimator_  # 最も精度がよいモデルを取得

print("Best Parameter : ", clf.best_params_)
print("Best Parameterでの検証用データの精度 : {:.2f}".format(clf.score(X_test, y_test)))
print("Best Parameterで交差検証した精度の平均（訓練データ） : {:.2f}".format(clf.best_score_))
```

    Best Parameter :  {'criterion': 'gini', 'max_depth': 3, 'max_leaf_nodes': 8}
    Best Parameterでの検証用データの精度 : 0.93
    Best Parameterで交差検証した精度の平均（訓練データ） : 0.95
    


```python
'''
可視化
'''
# tree_append.pngファイルを可視化
try:
    from pydotplus import graph_from_dot_data
    
    dot_data_append = tree.export_graphviz(
        best_clf, 
        out_file=None,
        feature_names=["petal_length [cm]", "petal_width [cm]"],
        class_names=["Setosa", "Versicolor", "Virginica"],
        filled=True, # Trueにすると、分岐の際にどちらのノードに多く分類されたのか色で示してくれる
        rounded=True, # Trueにすると、ノードの角を丸く描画する。
        impurity=True # Trueにすると、不純度を表示する。
    )

    graph = graph_from_dot_data(dot_data_append)
    graph.write_png('tree_append.png')
    
except ImportError as ex:
    print("Error: the pydotplus library is not installed.")

```


```python
Image(filename="./tree_append.png", width=800)
```




![png](output_41_0.png)




```python
# テストデータの強調表示なしでも表示できるかをテスト

# 分類結果の表示 
plot_decision_regions(X, y, classifier=best_clf)
# がく片の長さ
plt.xlabel("sepal length [cm]")
# 花びらの長さ
plt.ylabel("petal length [cm]")
# 凡例の設定(左上に配置)
plt.legend(loc="upper left")
# グラフの位置サイズの自動調整
plt.tight_layout()
```


![png](output_42_0.png)



```python
# 訓練データとテストデータの特徴量を行方向に結合
X_combined = np.vstack((X_train, X_test))
# 訓練データとテストデータのクラスラベルを結合
# 水平方向に(horizontal)連結
y_combined = np.hstack((y_train, y_test))

# 訓練データとテストデータ結合後の変数の中からtestデータの開始地点と終了地点を取得
test_str_pt = np.bincount(y).sum() - np.bincount(y_test).sum()
test_fin_pt = np.bincount(y).sum()

# 決定境界のプロット
plot_decision_regions(X=X_combined, y=y_combined, classifier=best_clf, test_idx=range(test_str_pt, test_fin_pt))

# 軸のラベルの設定
# 花びらの長さ
plt.xlabel("petal length [cm]")
# 花びらの幅
plt.ylabel("petal width [cm]")

# 凡例の設定(左上に配置)
plt.legend(loc="upper left")

# グラフを表示
plt.tight_layout()
plt.show()
```


![png](output_43_0.png)



```python
# 各特徴量の重要度を可視化
# (どの特徴量がどれくらい分割に寄与したのかを確認する)
n_features = X.shape[1] # 全説明変数

plt.barh(range(n_features), best_clf.feature_importances_, align="center") # 描画する際の枠組みを設定
plt.yticks(np.arange(n_features), ["petal_length [cm]", "petal_width [cm]"]) # 縦軸の設定
plt.show()

print("petal_length [cm]:", best_clf.feature_importances_[0])
print("petal_width [cm] :", best_clf.feature_importances_[1])
```


![png](output_44_0.png)


    petal_length [cm]: 0.5771640640061693
    petal_width [cm] : 0.4228359359938307
    


```python
print("決定木の正解率 : {:.2f}".format(best_clf.score(X_test, y_test)))
```

    決定木の正解率 : 0.93
    


```python
# 交差検証
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

tree_score = cross_val_score(best_clf, X, y, cv=kfold)

print(tree_score)
print(tree_score.mean())
```

    [1.         0.96666667 0.93333333 0.9        0.93333333]
    0.9466666666666667
    


```python
# テストデータで予測を行う
y_pred_best = best_clf.predict(X_test)

print("誤分類したデータ数: %d"% (y_test != y_pred_best).sum())
```

    誤分類したデータ数: 2
    


```python
# 予測結果の確認
print("決定木の正解率 : {:.2f}".format(best_clf.score(X_test, y_test)))
# マイクロ平均（micro）かマクロ平均（macro）
print("決定木の適合率 : {:.2f}".format(precision_score(y_test, y_pred_best, average="micro")))
# averageにマイクロ平均（micro）を設定
print("決定木の再現率 : {:.2f}".format(recall_score(y_test, y_pred_best, average="micro")))
# averageにマイクロ平均（micro）を設定
print("決定木のF値    : {:.2f}".format(f1_score(y_test, y_pred_best, average="micro")))
```

    決定木の正解率 : 0.93
    決定木の適合率 : 0.93
    決定木の再現率 : 0.93
    決定木のF値    : 0.93
    


```python

```
