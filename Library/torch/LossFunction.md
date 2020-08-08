# オプティマイザー（最適化用オブジェクト）の作成

### 書式

```python
import torch.nn as nn

# 変数
criterion = nn.MSELoss()   # 損失関数：平均二乗和誤差
```

### 引数

- `nn.MSELoss`も含めて以外が使用可能です
  - L1Loss（MAE：Mean Absolute Error、平均絶対誤差）
  - MSELoss（MSE：Mean Squared Error、平均二乗誤差）
  - CrossEntropyLoss（交差エントロピー誤差： クラス分類）
  - CTCLoss
  - NLLLoss
  - PoissonNLLLoss
  - KLDivLoss
  - BCELoss
  - BCEWithLogitsLoss
  - MarginRankingLoss
  - HingeEmbeddingLoss
  - MultiLabelMarginLoss
  - SmoothL1Loss
  - SoftMarginLoss
  - MultiLabelSoftMarginLoss
  - CosineEmbeddingLoss
  - MultiMarginLoss
  - TripletMarginLoss


### 説明

バックプロパゲーションで必要となる**損失関数（Loss Function）**を定義します。
損失関数は、自分で実装することも可能ですが、PyTorchに用意されているクラスをインスタンス化するだけで使用できます