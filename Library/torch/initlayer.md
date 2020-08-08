# 特定層の初期化関数

### 書式

```python
import torch.nn as nn

def init_parameters(layer):
    """パラメータ（重みとバイアス）の初期化
    引数の層が全結合層の時パラメータを初期化する
    
    Param:
      layer: 層情報
    """
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)    # 重みを「一様分布のランダム値」で初期化
        layer.bias.data.fill_(0.0)               # バイアスを「0」で初期化

# モデルのインスタンス化
model = NeuralNetwork()

model.apply(init_parameters)        # 学習の前にパラメーター初期化

```

### 説明

レイヤー（層）の種類が「nn.Linear」（全結合層）である場合には、重みをnn.init.xavier_uniform_()関数により「一様分布のランダム値」に初期化、バイアスをlayer.bias.data.fill_(0.0)メソッドにより「0」に初期化する関数です