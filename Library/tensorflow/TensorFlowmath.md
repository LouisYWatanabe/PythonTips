# TensorFlowの演算子

| 演算子            | 説明                                           |
|-------------------|------------------------------------------------|
| tf.add(x, y)      | 同じ型のTensorを足し算します。(x+y)            |
| tf.sybtract(x, y) | 同じ型のTensorを引き算します。(x-y)            |
| tf.multiply(x, y) | 2つのTensorに対して行列の積を求めます。        |
| tf.pow(x, y)      | xの要素ごとにy乗します。                       |
| tf.exp(x)         | pow(e, x)を行う。eはネイピア数                 |
| tf.sqrt(x)        | xの平方根を求めます。pow(x, 0.5)と等価です。   |
| tf.div(x, y)      | xとyの要素ごとの除算を行います。               |
| tf.truediv(x, y)  | float型の引数をキャストする以外は、div()と同じ |
| tf.floordiv(x, y) | 求めた値を整数に丸める以外は、div()と同じ      |
| tf.mod(x, y)      | 要素ごとの剰余                                 |