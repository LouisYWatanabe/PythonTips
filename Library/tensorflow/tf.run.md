# Sessionオブジェクトの作製

### 書式

```python
tf.run(
	fetches,
	feed_dict=None,
	options=None,
	run_metadata=None
)
```

### 引数

- fetches	計算グラフのopノード
- feed_dict	型の指定
- options	オプション。トレースをオンにするなどステップの振る舞いを制御できる
- run_metadata	オプション。非Tensorオブジェクトからのしょつ力を収集します

### 例

```python
import tensorflow as tf

# データフローグラフの作成
rate = tf.constant(0.08, dtype=tf.float32)    # 消費税
price = tf.Variable(100, dtype=tf.float32)    # 価格を保持する変数
tax = tf.multiply(rate, price)                # 税額を計算するopノード
price_tax = tf.add(price, tax)                # 税込み金額を計算するopノード
update = tf.assign(price, price_tax)          # 変数priceを更新するopノード

# データフローグラフの作成
rate = tf.constant(0.08, dtype=tf.float32)        # 消費税
price = tf.Variable(100, dtype=tf.float32)        # 価格を保持する変数
tax = tf.multiply(rate, price)                    # 税額を計算するopノード
price_tax = tf.add(price, tax)                    # 税込み金額を計算するopノード
update = tf.assign(price, price_tax)              # 変数priceを更新するopノード
# セッションの処理
with tf.Session() as sess:
    sses.run(tf.global_variables_initializer())   # すべての変数をまとめて初期化
    print('rate          = ', sess.run(rate))     # rateノードを実行して出力
    print('price         = ', sess.run(price))    # 変数priceを出力
    sess.run(update)                              # updateノードを実行
    print('price(update) = ', sess.run(price))    # 更新後の変数priceを出力
```
	rate          =  0.08
	price         =  100.0
	price(update) =  108.0
```python
import tensorflow as tf

# データフローグラフの作成
rate = tf.constant(0.08, dtype=tf.float32)    # 消費税
price = tf.Variable(100, dtype=tf.float32)    # 価格を保持する変数
tax = tf.multiply(rate, price)                # 税額を計算するopノード
price_tax = tf.add(price, tax)                # 税込み金額を計算するopノード
update = tf.assign(price, price_tax)          # 変数priceを更新するopノード

# データフローの各要素を出力
print('rate          = ', rate)
print('price         = ', price)
print('tax           = ', tax)
print('price_tax     = ', price_tax)
print('update        = ', update)

# セッション部
sess = tf.Session()                           # セッションスタート
print('rate          = ', sess.run(rate))     # rateノードを実行して出力
sess.run(price.initializer)                   # 変数priceは初期化が必要
print('price         = ', sess.run(price))    # 変数priceを出力

sess.run(update)                              # updateノードを実行
print('price(update) = ', sess.run(price))    # 更新後の変数priceを出力

sess.close()                                  # セッションを閉じる
```
	rate          =  Tensor("Const_1:0", shape=(), dtype=float32)
	price         =  <tf.Variable 'Variable_1:0' shape=() dtype=float32_ref>
	tax           =  Tensor("Mul_1:0", shape=(), dtype=float32)
	price_tax     =  Tensor("Add_1:0", shape=(), dtype=float32)
	update        =  Tensor("Assign_1:0", shape=(), dtype=float32_ref)
	rate          =  0.08
	price         =  100.0
	price(update) =  108.0

### 説明
