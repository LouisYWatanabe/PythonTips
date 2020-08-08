
ResNetを実装し、内容を確認します。
事前に`ResidualBlock`モデルを作成し、それを何層も重ねる形でResNetを実装します。


```python
import torch
import torch.nn as nn

# 定数
KERNEL_SIZE = 3             # 入力層のフィルター(カーネル) サイズ 3×3のサイズのカーネルを作成
PADDING_SIZE1 = 1           # パディングサイズ
PADDING_SIZE0 = 0           # パディングサイズ
# 入力のチャネル数を出力のチャネル数に変化させるためにkernel_size = 1を定義
KERNEL_SIZE1 = 1            # 入力層のフィルター(カーネル) サイズ 1×1のサイズのカーネルを作成

# モデルの定義
# モデルをカスタムするのでnn.Sequential()は使用せずforward()で設計する
class ResidualBlock(nn.Module):
    def __init__(self, INPUT_CHANNELS, OUTPUT_CHANNELS):
        """
        Param:
            INPUT_CHANNELS:  入力チャネル数
            OUTPUT_CHANNELS: 出力チャネル数
        """
        super(ResidualBlock, self).__init__()
        
        """残差ブロックの定義
        """
        # 畳み込み層1
        self.conv1 = nn.Conv2d(
            INPUT_CHANNELS,                      # 入力チャネル数
            OUTPUT_CHANNELS,                     # 出力チャネル数
            kernel_size = KERNEL_SIZE,           # 入力層のフィルター(カーネル) サイズ
            padding = PADDING_SIZE1              # 入力層のパディングサイズ
        )
        # 畳み込み層2
        self.conv2 = nn.Conv2d(
            # 出力チャネル数を受け取り出力チャネル数を返す層
            OUTPUT_CHANNELS,                     # 入力チャネル数
            OUTPUT_CHANNELS,                     # 出力チャネル数
            kernel_size = KERNEL_SIZE,           # 入力層のフィルター(カーネル) サイズ
            padding = PADDING_SIZE1              # 入力層のパディングサイズ
        )
        # 畳み込み層3
        self.conv3 = nn.Conv2d(
            # 入力のチャネル数を出力のチャネル数に変化させるためにkernel_size = 1を使用
            INPUT_CHANNELS,                      # 入力チャネル数
            OUTPUT_CHANNELS,                     # 出力チャネル数
            kernel_size = KERNEL_SIZE1,          # 入力層のフィルター(カーネル) サイズ
            padding = PADDING_SIZE0              # 入力層のパディングサイズ
        )
        # バッチノーマライゼーションの定義
        self.bn = nn.BatchNorm2d(
            OUTPUT_CHANNELS,                     # チャネル数
        )
        # 活性化関数 ReLU関数
        self.relu = nn.ReLU()
        
    def shortcut(self, x):
        """identityの入力チャネル数を合わせる
        浅い層のモデルを構築し、
        スキップコネクションする
        Param:
            x: identity
        """
        x = self.conv3(x)                  # 出力をOUTPUT_CHANNELSにする
        x = self.bn(x)                     # バッチノーマライゼーション
        return x
        
    def forward(self, x):
        """フォワードパスの定義
        """
        # 出力＝活性化関数（第n層（入力））の形式
        identity = x                       # 恒等関数の定義
        # 一回目の畳み込み
        x = self.conv1(x)                  # 畳み込み層1
        x = self.bn(x)                     # バッチノーマライゼーション
        x = self.relu(x)                   # 活性化関数 ReLU関数
        # 二回目の畳み込み
        x = self.conv2(x)                  # 畳み込み層2
        x = self.bn(x)                     # バッチノーマライゼーション
        x += self.shortcut(identity)       # スキップコネクションの追加 チャネル数を合わせる（入力がINPUT_CHANNELSなのでOUTPUT_CHANNELSに変えないとエラーになる）
        return x
```


```python
# 定数
LAYER_NEURONS = 28*28*64      # 入力ユニット数 28×28の画像が入力され、out_channelsが64だと仮定します
OUTPUT_RESULTS = 10           # 出力結果への出力ユニット数　10クラスの分類に使用すると仮定します

# モデルの定義
# モデルをカスタムするのでnn.Sequential()は使用せずforward()で設計する
class ResNet(nn.Module):
    def __init__(self, block):
        """
        Param:
            block: ResidualBlock()を渡す変数
        """
        # 継承したnn.Module親クラスを初期化
        super(ResNet, self).__init__()
        
        # 出力層
        self.layer_out = nn.Linear(
            in_features = LAYER_NEURONS,        # 入力ユニット数
            out_features = OUTPUT_RESULTS       # 出力結果への出力ユニット数
        )
        # 積み上げる層の定義
        self.make_layer = self._make_layer(
            block,
            3,                                  # 積み上げる層の数
            3,                                  # 画像を想定して3チャネル
            64                                  # 出力チャネル数は64と仮定します
        )
    
    def _make_layer(self, block, num_residual_blocks, in_channels, out_channels):
        """
        Param:
            block: ResidualBlock()
            num_residual_blocks: 何層作るのか
            in_channels: 入力チャネル数
            out_channels: 出力チャネル数
        """
        layers = []        # 積み上げた層の格納先
        for i in range(num_residual_blocks):
            """
            層ごとのチャネル数の設定
            
            入力層:
                入力チャネル数: in_channels
                出力チャネル数: out_channels
            入力層以外:
                入力チャネル数: out_channels
                出力チャネル数: out_channels
            """
            if i == 0:
                # 入力層
                layers.append(block(in_channels, out_channels))
            else:
                # 入力層以外
                layers.append(block(out_channels, out_channels))
        # 複数のブロックを積み上げたモデル返す
        return nn.Sequential(*layers)
        
    def forward(self, x):
        """フォワードパスの定義
        """
        # 出力＝活性化関数（第n層（入力））の形式
        identity = x                       # 恒等関数の定義
        # 一回目の畳み込み
        x = self.make_layer(x)             # 積み上げる層
        x = x.view(-1, LAYER_NEURONS)      # 層の出力サイズを全結合層の入力ニューロン数に変換
        # x = x.view(x.size(0), -1)  # (チャネル数, -1)層の出力サイズを全結合層の入力ニューロン数に変換
        x = self.layer_out(x)              # 出力層
        return x

# モデルのインスタンス化
model = ResNet(ResidualBlock)
print(model)                      # モデルの概要を出力
```

    ResNet(
      (layer_out): Linear(in_features=50176, out_features=10, bias=True)
      (make_layer): Sequential(
        (0): ResidualBlock(
          (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv3): Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
        (1): ResidualBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
        (2): ResidualBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU()
        )
      )
    )
    

## 順伝播の実行


```python
x_test = torch.randn(32, 3, 28, 28)     # ミニバッチ数32、チャネル数3、画像サイズ28×28
out_put = model(x_test)                 # 順伝播の実行
out_put.size()                          # 結果のサイズを表示
```




    torch.Size([32, 10])


