# ドロップアウトの適用

### 書式

```python
nn.Dropout(dropout_probability)
```

### 引数
dropout_probability: ドロップアウト発生確率（0.5以上はないらしい）
### 例


```python
import torch.nn as nn
# 定数
INPUT_CHANNELS = 1         # 入力チャネル数（MNISTはRGB値のようなカラー画像ではなく、各ピクセルが0～255（の値を-1～1の範囲の浮動小数点数に変換したもの）だけのデータなので、ここでは1を指定）
CONV2_CHANNELS = 6         # 出力チャネル数 6個のカーネルを作成（conv2の入力チャネルの数を一致させる必要があります[conv1の出力→活性化関数→プーリングを経てconv2の入力チャンネルとして渡されます]）
INPUT_KERNELSIZE = 5  # 入力層のフィルター(カーネル) サイズ 5×5のサイズのカーネルを作成
OUTPUT_CHANNELS = 16       # 畳み込み層2の出力チャネル数

# 2×2のサイズでプーリングを行います
INPUT_POOLSIZE = (2, 2)    # プーリング実行サイズ 2×2のサイズ
# PADDING_SIZE = 0           # パディングサイズ ゼロパディング
# STRIDE_WIDTH = 1           # ストライド幅
DROPOUT_PROBABILITY = 0.5    # ドロップアウト確率

LAYER1_NEURONS = 16 * 16   # 隠れ層1のニューロン数 256個はreshapeで作成
LAYER2_NEURONS = 64        # 隠れ層2のニューロン数

OUTPUT_RESULTS = 10        # 出力層のニューロン数

# 変数 活性化関数
activation = torch.nn.ReLU()     # 活性化関数（隠れ層）ReLU関数    変更可

# モデルの定義
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        
        """層の定義
        """
        # 畳み込み層1
        self.conv1 = nn.Conv2d(
            INPUT_CHANNELS,       # 入力チャネル数
            CONV2_CHANNELS,       # 出力チャネル数
            INPUT_KERNELSIZE      # 入力層のフィルター(カーネル) サイズ
        )
        # プーリング層（MaxPooling）
        self.pool = nn.MaxPool2d(
            INPUT_POOLSIZE,       # プーリング実行サイズ
            # padding=PADDING_SIZE, # パディングサイズ
            # stride=STRIDE_WIDTH   # ストライド幅
        )
        # 畳み込み層2
        self.conv2 = nn.Conv2d(
            CONV2_CHANNELS,       # 入力チャネル数
            OUTPUT_CHANNELS,      # 出力チャネル数
            INPUT_KERNELSIZE      # 入力層のフィルター(カーネル) サイズ
        )
        
        # 隠れ層1
        self.layer1 = nn.Linear(
            LAYER1_NEURONS,      # 入力層のユニット数
            LAYER2_NEURONS       # 次の層への出力ユニット数
        )
        # ドロップアウト
        self.dropout = nn.Dropout(
            DROPOUT_PROBABILITY  # ドロップアウトの確率
        )
        # 出力層
        self.layer_out = nn.Linear(
            LAYER2_NEURONS,      # 入力ユニット数
            OUTPUT_RESULTS       # 出力結果への出力ユニット数
        )
        
    def forward(self, x):
        """フォワードパスの定義
        """
        # 出力＝活性化関数（第n層（入力））の形式
        x = activation(self.conv1(x))      # 活性化関数は変数として定義
        x = self.pool(x)                   # マックスプーリングの実行
        x = self.conv2(x)
        x = self.pool(x)
        x = x.reshape(-1, LAYER1_NEURONS)  # 畳み込み層の出力サイズを全結合層の入力ニューロン数に変換
        x = activation(self.layer1(x))     # 活性化関数は変数として定義
        x = self.dropout(x)                # ドロップアウト層
        x = self.layer_out(x)              # 出力層の実行
        return x

# モデルのインスタンス化
model = ConvolutionalNeuralNetwork()
print(model)                      # モデルの概要を出力  
```

	ConvolutionalNeuralNetwork(
		(conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
		(pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
		(conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
		(layer1): Linear(in_features=256, out_features=64, bias=True)
		(dropout): Dropout(p=0.5, inplace=False)
		(layer_out): Linear(in_features=64, out_features=10, bias=True)
	)

### 説明