# データローダーの作成(ミニバッチ学習の使用)

### 書式

```python
from torch.utils.data import DataLoader, TensorDataset    # データ関連のユーティリティクラスのインポート

# 定数（学習方法の設計時）
BATCH_SIZE = 15        # バッチサイズ：15

# データセット（dataset）の作成 入力データ（X）と教師ラベル（y）をデータセットにまとめる
dt_train = TensorDataset(t_X_train, t_y_train)    # 訓練データ
dt_test = TensorDataset(t_X_test, t_y_test)       # テストデータ精度検証

# データローダー（loader）の作成 ミニバッチを扱うため、データローダー（loader）を作成する
loader_train = DataLoader(dt_train, batch_size=BATCH_SIZE, shuffle=True)
loader_test = DataLoader(dt_test, batch_size=BATCH_SIZE)
```

### 引数

`DataLoader`クラス
- `shuffle` データをシャッフルするか（**True**）しないか（**False**）を指定できます。


### 説明

PyTorchには、**ミニバッチ学習**を簡単扱うための`DataLoader`クラスが用意されています。
このクラスを利用するには、既存のデータや教師ラベルといった**テンソルを一つの**`TensorDataset`にまとめる必要があります。