

```python
# XORの学習
from keras.models import Sequential     # 初期化メソッド
from keras.layers import Dense          # 層を表現するメソッド
from keras.optimizers import SGD
import numpy as np
```


```python
# 訓練データ
train = np.array([
                  [0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]
])

# 正解ラベル
label = np.array([
                  [0],
                  [1],
                  [1],
                  [0]
])
```

Sequentialオブジェクトを生成して隠れ層を追加する

Dense()は、層を表現するDenceオブジェクトを生成します

```
Dense(
        units=隠れ層のニューロン数,
        input_dim=入力されるニューロン数,
        activation='活性化関数の種類'
)
```


```python
# Sequentialオブジェクトを生成
model = Sequential()

# (第1層)隠れ層
model.add(
    Dense(
        units=2,                # 隠れ層のニューロン数は2
        input_dim=2,            # 入力層のニューロン数は2
        activation='sigmoid'    # 活性化関数はシグモイド
    )
)
# (第2層)出力層
model.add(
    Dense(
        units=1,                # 出力層のニューロン数は1
        activation='sigmoid'    # 活性化関数はシグモイド
    )
)
```

compile()メソッドでSequentialオブジェクトをコンパイルします


```python
model.compile(
    loss='binary_crossentropy', # 誤差関数にバイナリ用のクロスエントロピーを指定
    optimizer=SGD(lr=0.1),      # 勾配降下法を指定
)
```


```python
# 作成したニューラルネットの概要を出力する
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 2)                 6         
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 3         
    =================================================================
    Total params: 9
    Trainable params: 9
    Non-trainable params: 0
    _________________________________________________________________

学習の実行にはfit()メソッドを使い、<br>
batch_sizeは確率的勾配降下法における勾配計算に使用する学習データの数（サンプル数）を指定します



```python
train.shape[0]
```




    4




```python
# 学習の実行
history = model.fit(
    train,                          # 訓練データ
    label,                          # 正解ラベル
    epochs=3000,                    # 学習回数
    batch_size=train.shape[0],      # ミニバッチのサイズ（すべての訓練データを使用）
    verbose=0,                      # 学習ごとの進捗状況を出力しない
)

```


```python
# 学習結果の確認
# predict_classes()で出力のみを行う
# 0.5を閾値として0または1を取得
classes = model.predict_classes(train, batch_size=train.shape[0])
# 出力された値そのものを取得
prob = model.predict_proba(train, batch_size=train.shape[0])

print('Output:')
print(classes)

print('Output Probability:')
print(prob)
```

    Output:
    [[1]
     [1]
     [0]
     [0]]
    Output Probability:
    [[0.50327986]
     [0.5183543 ]
     [0.48312703]
     [0.4953884 ]]
    


