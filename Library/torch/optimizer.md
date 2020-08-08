# オプティマイザー（最適化用オブジェクト）の作成

### 書式

```python
import torch.optim as optim    # 最適化モジュールのインポート

# 定数
LEARN_RATE = 0.03        # 学習率
REGULAR =  0.03          # 正則化率

# オプティマイザーの作成（パラメータと学習率も作成）
optimizer = optim.SGD(    # 最適化アルゴリズムに「SGD」を選択
    model.parameters(),   # 最適化で更新する重みやバイアスのパラメータ
    lr=LEARN_RATE,        # 学習率
    weight_decay=REGULAR  # L2正則化（不要なら省略）
)
```

### 引数

- `torch.optim.SGD`を含めて以下が使用可能です
  - Adadelta
  - Adagrad
  - Adam（有名）
  - AdamW
  - SparseAdam
  - Adamax
  - ASGD
  - LBFGS
  - RMSprop
  - Rprop
  - SGD（確率的勾配降下法）
- パラメーター： 重みやバイアスを、＜モデル名＞.parameters()メソッドで取得し指定します。
- lr： 学習率。この例では定数LEARNING_RATEとして定義しています
- weight_decay： 正則化率。正則化（Regularization）は「L2」（＝重み減衰： Weight Decay）に相当します（※あまり使わない「L1」はPyTorchによる最適化アルゴリズムの基本機能には含まれていない）


### 説明

学習を行う**最適化アルゴリズム（Optimizer：オプティマイザ）**は、自分で実装することも可能ですが、PyTorchに用意されているクラスをインスタンス化するだけで使用できます