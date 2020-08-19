# TensorFlow

> 機械学習用モジュール
プログラムで行いたい処理をデータフローグラフとしてまとめます

- [TensorFlowの演算子](./tensorflow/TensorFlowmath.md)
- [tf.constant 定数の値を保持する](./tensorflow/tf.constant.md)
- [tf.Variable 変数の値を保持する](./tensorflow/tf.variable.md)
- [tf.truncated_normal 正規分布ランダムサンプリング](./tensorflow/tf.truncatednormal.md)
- [tf.placeholder データの保持](./tensorflow/tf.placeholder.md)
- [tf.Session Sessionオブジェクトの生成](./tensorflow/tf.session.md)
- [tf.run opノードを評価し、operationの結果を返す](./tensorflow/tf.run.md)
- [tf.train.GradientDescentOptimizer 勾配降下アルゴリズム](./tensorflow/tf.train.GradientDescentOptimizer.md)
- [tf.matmul 行列の掛算](./tensorflow/tf.matmul.md)

- TensorFlowにおけるモデル作成の流れ
    - [全結合層によるモデル作成の流れ](./tensorflow/tf2_base/tf2_base.md)
    - Sequential（積層型）モデル： コンパクトで簡単な書き方
        - [Sequential（積層型）モデルによるコンパクトで簡単な書き方](./tensorflow/Sequential1/Sequential1.md)
        - [Sequentialオブジェクトのaddメソッドで追加［tf.keras - Sequential API］](./tensorflow/Sequential2/Sequential2.md)
    - Functional（関数型）API： 複雑なモデルも定義できる柔軟な書き方
        - [Modelクラスのコンストラクター利用［tf.keras - Functional API］](./tensorflow/Functional/Functional.md)
        - [Modelクラスのサブクラス化を利用した書き方](./tensorflow/Functional2/Functional2.md)
