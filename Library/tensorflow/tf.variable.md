# 変数の値を保持する

### 書式

```python
tf.Variable(
	initial_value=None,
	trainable=True,collections=None, catching_device=None, name=None,
	variable_def=None,
	dtype=None,
	expected_shape=None,
	import_scope=None,
	constraint=None
)
```

### 引数

- initial_value	変数の初期値を指定します。
- trainable	Trueであれば、Optimizerで使用されるGraphKeys.TRAINABLE_VARIABLESに追加されます。
- collections	グラフのCollection Keyリスト、デフォルトは[GraphKeys.GLOBAL_VARIABLES]。
- validate_shape	bool	Falseにすると、型や形状チェックしなくなります。
- caching_device	変数を読み込む際に、キャッシュするデバイスを指定します。
- name	変数の名前。デフォルトでは自動でユニークな名前を割り当てます。
- validable_def	VariableDefプロトコルバッファ。Noneでなければ、validable_defに合わせて再作成します。
- dtype	指定されていれば、dtypeに合わせて初期値が変換されます。
- expected_shape	指定されていれば、expected_shapeに合わせて形状変換されます。
- import_scope	追加する名前空間。
- constraint	更新された後に変数に適用される関数を指定するオプション

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
