# 行列の掛け算

### 書式

```python
a = tf.Variable([[1, 2, 3],
                 [3, 4, 5]])
y = tf.matmul(a, [[1],
                  [2],
                  [3]])
```
	[[14]
 	[26]]

### 引数

### 例

```python
# GoogleColab用のコード
# 現在インストールされているバージョンの確認

!pip show tensorflow
```


```python
# # GoogleColab用のコード
# ライブラリの特定のバージョンへ変更する

!pip install tensorflow==1.3
```

#### TensorFlowでAND回路定義する


```python
import numpy as np
import tensorflow as tf

# ANDゲート
# x1, x2を行列xに代入(4, 2)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# 正解値をtに代入(4, 1)
T = np.array([[0], [0], [0], [1]])
```


```python
learn_rate = 0.1                # 学習率
train_epoch = 100               # 学習回数
```


```python
"""
データフローグラフ
"""
# 重み行列(2, 1)正規分布からランダムサンプリング
# 切断正規分布の母集団からランダムサンプリング
w = tf.Variable(tf.truncated_normal([2, 1]))
# バイアスは0で初期化
b = tf.Variable(tf.zeros([1]))

# 入力データを保持するブレースホルダー 形状（行数未定, 2列）
x = tf.placeholder(tf.float32, shape=[None, 2])
# 正解データを保持するブレースホルダー 形状（行数未定, 1列）
t = tf.placeholder(tf.float32, shape=[None, 1])

# シグモイド関数の出力を行うopノード
out = tf.nn.sigmoid(tf.matmul(x, w) + b)
# 誤差関数（交差エントロピー誤差）のopノード
cost = - tf.reduce_sum(
    t*tf.log(out) + (1 - t)*tf.log(1-out)
)
# 勾配効果アルゴリズムのopノード
train_op = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)
```


```python
# テストデータの予測を判定するopノード
correct_prediction = tf.equal(tf.to_float(tf.greater(out, 0.5)), t)
# 変数を初期化するopノード
init_op = tf.global_variables_initializer()
```


```python
# 学習の実行
with tf.Session() as sess:
    sess.run(init_op)           # init_opノード実行して変数を初期化
    # train_epochの回数だけ学習を繰り返す
    for epoch in range(train_epoch):
        # cost誤差関数の戻り値をerrに代入、train_opは戻り値は使用しない
        err, _ = sess.run(
            [cost,              # 交差エントロピー誤差関数ノードを実行
            train_op],          # ミニマイザーを実行
            feed_dict={
                x: X,           # 訓練データ
                t: T            # 正解ラベル
            }
        )
        epoch += 1              # インクリメント
        if epoch%10 == 0:       # 10回
            print('{}回: Parameter = {}'.format(epoch, err))
    # 学習終了時の誤差を出力
    print('{}回: Parameter = {}'.format(epoch, err))

    # 学習結果で予測判定をする
    # 予測ができていることを確認
    print('\nCheck:')
    print(
        correct_prediction.eval(
            # 訓練データ、正解ラベルをセット
            feed_dict={x: X, t:T}
        )
    )

    # 学習後のバイアスと重みを出力
    print(
        '\nb = ', sess.run(b),
        '\nw1, w2 = ', sess.run(w), '\n'
    )
    # 最終出力を表示
    print('Output Probability:')
    print(
        out.eval(
            # out2ノード（活性化関数）を実行
            # 訓練データ
            feed_dict={x: X}
        )
    )
    print('\nOutput Probability (>= 0.5 == 1):')
    print(
        (out.eval(
            feed_dict={x: X}
        ) >= 0.5).astype(np.int)
    )
```

    10回: Parameter = 2.250655174255371
    20回: Parameter = 1.9506781101226807
    30回: Parameter = 1.7355761528015137
    40回: Parameter = 1.5679242610931396
    50回: Parameter = 1.433485507965088
    60回: Parameter = 1.3231124877929688
    70回: Parameter = 1.2306122779846191
    80回: Parameter = 1.1517117023468018
    90回: Parameter = 1.0833990573883057
    100回: Parameter = 1.023504614830017
    100回: Parameter = 1.023504614830017
    
    Check:
    [[ True]
     [ True]
     [ True]
     [ True]]
    
    b =  [-2.9818609] 
    w1, w2 =  [[1.8862725]
     [1.7061484]] 
    
    Output Probability:
    [[0.0482521 ]
     [0.21828094]
     [0.25056744]
     [0.64806855]]
    
    Output Probability (>= 0.5 == 1):
    [[0]
     [0]
     [0]
     [1]]
    


```python
# NANDゲート
# x1, x2を行列xに代入(4, 2)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# 正解値をtに代入(4, 1)
T = np.array([[1], [1], [1], [0]])

learn_rate = 0.1                # 学習率
train_epoch = 100               # 学習回数

"""
データフローグラフ
"""
# 重み行列(2, 1)正規分布からランダムサンプリング
# 切断正規分布の母集団からランダムサンプリング
w = tf.Variable(tf.truncated_normal([2, 1]))
# バイアスは0で初期化
b = tf.Variable(tf.zeros([1]))

# 入力データを保持するブレースホルダー 形状（行数未定, 2列）
x = tf.placeholder(tf.float32, shape=[None, 2])
# 正解データを保持するブレースホルダー 形状（行数未定, 1列）
t = tf.placeholder(tf.float32, shape=[None, 1])

# シグモイド関数の出力を行うopノード
out = tf.nn.sigmoid(tf.matmul(x, w) + b)
# 誤差関数（交差エントロピー誤差）のopノード
cost = - tf.reduce_sum(
    t*tf.log(out) + (1 - t)*tf.log(1-out)
)
# 勾配効果アルゴリズムのopノード
train_op = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

# テストデータの予測を判定するopノード
correct_prediction = tf.equal(tf.to_float(tf.greater(out, 0.5)), t)
# 変数を初期化するopノード
init_op = tf.global_variables_initializer()

# 学習の実行
with tf.Session() as sess:
    sess.run(init_op)           # init_opノード実行して変数を初期化
    # train_epochの回数だけ学習を繰り返す
    for epoch in range(train_epoch):
        # cost誤差関数の戻り値をerrに代入、train_opは戻り値は使用しない
        err, _ = sess.run(
            [cost,              # 交差エントロピー誤差関数ノードを実行
            train_op],          # ミニマイザーを実行
            feed_dict={
                x: X,           # 訓練データ
                t: T            # 正解ラベル
            }
        )
        epoch += 1              # インクリメント
        if epoch%10 == 0:       # 10回
            print('{}回: Parameter = {}'.format(epoch, err))
    # 学習終了時の誤差を出力
    print('{}回: Parameter = {}'.format(epoch, err))

    # 学習結果で予測判定をする
    # 予測ができていることを確認
    print('\nCheck:')
    print(
        correct_prediction.eval(
            # 訓練データ、正解ラベルをセット
            feed_dict={x: X, t:T}
        )
    )

    # 学習後のバイアスと重みを出力
    print(
        '\nb = ', sess.run(b),
        '\nw1, w2 = ', sess.run(w), '\n'
    )
    # 最終出力を表示
    print('Output Probability:')
    print(
        out.eval(
            # out2ノード（活性化関数）を実行
            # 訓練データ
            feed_dict={x: X}
        )
    )
    print('\nOutput Probability (>= 0.5 == 1):')
    print(
        (out.eval(
            feed_dict={x: X}
        ) >= 0.5).astype(np.int)
    )
```

    10回: Parameter = 1.9043970108032227
    20回: Parameter = 1.6423313617706299
    30回: Parameter = 1.492703914642334
    40回: Parameter = 1.373030662536621
    50回: Parameter = 1.273414134979248
    60回: Parameter = 1.1888914108276367
    70回: Parameter = 1.1160537004470825
    80回: Parameter = 1.0524576902389526
    90回: Parameter = 0.9963120222091675
    100回: Parameter = 0.9462747573852539
    100回: Parameter = 0.9462747573852539
    
    Check:
    [[ True]
     [ True]
     [ True]
     [ True]]
    
    b =  [3.2187607] 
    w1, w2 =  [[-2.018616 ]
     [-1.9097793]] 
    
    Output Probability:
    [[0.9615342 ]
     [0.78734267]
     [0.7685506 ]
     [0.3296796 ]]
    
    Output Probability (>= 0.5 == 1):
    [[1]
     [1]
     [1]
     [0]]
    


### 説明
