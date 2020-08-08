# 転移学習 

Kaggleの[Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)から`dogs-vs-cats-redux-kernels-edition.zip`をダウンロードし、VGG16で転移学習を行います。

読み込んだモデルの最後の出力だけを2に変更し、モデルパラメータは出力層(fc)のみを渡して学習を行います。

## データのロード

- dogs-vs-cats-redux-kernels-edition.zipを./data/dogs-vs-cats-redux-kernels-edition内に解凍
- ./data/dogs-vs-cats-redux-kernels-edition内のtrain.zipとtest.zipをその場で解凍



```python
# zipファイルの解凍
import os
import zipfile

dogs_cats_dir = './data/dogs-vs-cats-redux-kernels-edition/'

# フォルダ'./data/dogs-vs-cats-redux-kernels-edition.zip'が存在しない場合にzipの解凍
if not os.path.exists(dogs_cats_dir):
    # dogs-vs-cats-redux-kernels-edition.zipを解凍
    with zipfile.ZipFile('./data/dogs-vs-cats-redux-kernels-edition.zip','r') as file:
        # /dogs-vs-cats-redux-kernels-editionディレクトリを作りその中に解凍ファイルを作製
        file.extractall(dogs_cats_dir)

# フォルダ'./data/dogs-vs-cats-redux-kernels-edition/'内にtestディレクトリが存在しない場合、
# 訓練データとテストデータのzipを解凍
if not os.path.exists(dogs_cats_dir+'test/'):
    # 訓練データとテストデータのzipを解凍
    for f in ['train.zip', 'test.zip']:
        with zipfile.ZipFile(dogs_cats_dir + f,'r') as file:
            file.extractall(dogs_cats_dir)
```


```python
# 解凍したデータフォルダの構造を確認する
!ls ./data/dogs-vs-cats-redux-kernels-edition/
```

    logs  sample_submission.csv  test  test.zip  train  train.zip  valid



```python
from glob import glob

# 全ての.jpgファイルパスのリストを取得
g_train = glob('./data/dogs-vs-cats-redux-kernels-edition/train/*.jpg')
g_test = glob('./data/dogs-vs-cats-redux-kernels-edition/test/*.jpg')
# データ数の確認
print('train image size', len(g_train))
print('test image size', len(g_test))
```

    train image size 0
    test image size 0


### 訓練データの中からバリデーションデータを抽出

訓練データからランダムに選んだ2000枚をバリデーションデータとします。


```python
import numpy as np

valid_path = './data/dogs-vs-cats-redux-kernels-edition/valid/'
# validディレクトリ
if not os.path.exists(valid_path):
    os.mkdir(valid_path)
    # ファイルパスをリストで取得
    g = glob('./data/dogs-vs-cats-redux-kernels-edition/train/*.jpg')
    
    # 訓練データからランダムに選んだ2000枚をバリデーションデータとします
    shuf = np.random.permutation(g)
    for i in range(2000):
        # ファイル名から./data/dogs-vs-cats-redux-kernels-edition/train/`の部分を取り除きファイル名にする
        # trainからvalidのディレクトリに画像ファイルを移動
        os.rename(shuf[i], os.path.join(valid_path, shuf[i][48:]))

```


```python
# 全ての.jpgファイルパスのリストを取得
g_train = glob('./data/dogs-vs-cats-redux-kernels-edition/train/*.jpg')
g_valid = glob('./data/dogs-vs-cats-redux-kernels-edition/valid/*.jpg')
g_test = glob('./data/dogs-vs-cats-redux-kernels-edition/test/*.jpg')
# データ数の確認
print('train image size', len(g_train))
print('valid image size', len(g_valid))
print('test image size', len(g_test))
```

    train image size 0
    valid image size 0
    test image size 0


### クラスごとにサブディレクトリを作成する

PyTorchで読み込みやすいようにクラスごとにサブディレクトリを作成します。<br>Kaggleのテストデータは正解ラベルがついていないため unknown というサブディレクトリに入れます。

最終的なデータの階層構造は

```
train
  cats
    cat.1.jpg
    cat.2.jpg
  dogs
    dog.1.jpg
    dog.2.jpg
valid
  cats
    cat.10.jpg
    cat.21.jpg
  dogs
    dog.8.jpg
    dog.52.jpg
test
  unknown
    1.jpg
    2.jpg
```


```python
g_train[:2]
```




    []




```python
# 訓練データ
# 全ての.jpgファイルパスのリストを取得
g_train_dog = glob('./data/dogs-vs-cats-redux-kernels-edition/train/dog*.jpg')
g_train_cat = glob('./data/dogs-vs-cats-redux-kernels-edition/train/cat*.jpg')

train_dog_path = './data/dogs-vs-cats-redux-kernels-edition/train/dog/'
train_cat_path = './data/dogs-vs-cats-redux-kernels-edition/train/cat/'

# dogディレクトリの処理
if not os.path.exists(train_dog_path):
    os.mkdir(train_dog_path)
    for g in range(len(g_train_dog)):
        # ファイル名から./data/dogs-vs-cats-redux-kernels-edition/train/`の部分を取り除きファイル名にする
        # trainからvalidのディレクトリに画像ファイルを移動
        os.rename(g_train_dog[g], os.path.join(train_dog_path, g_train_dog[g][48:]))
# catディレクトリの処理
if not os.path.exists(train_cat_path):
    os.mkdir(train_cat_path)
    for g in range(len(g_train_cat)):
        # ファイル名から./data/dogs-vs-cats-redux-kernels-edition/train/`の部分を取り除きファイル名にする
        # trainからvalidのディレクトリに画像ファイルを移動
        os.rename(g_train_cat[g], os.path.join(train_cat_path, g_train_cat[g][48:]))

```


```python
# バリデーションデータ
# 全ての.jpgファイルパスのリストを取得
g_valid_dog = glob('./data/dogs-vs-cats-redux-kernels-edition/valid/dog*.jpg')
g_valid_cat = glob('./data/dogs-vs-cats-redux-kernels-edition/valid/cat*.jpg')

valid_dog_path = './data/dogs-vs-cats-redux-kernels-edition/valid/dog/'
valid_cat_path = './data/dogs-vs-cats-redux-kernels-edition/valid/cat/'

# dogディレクトリの処理
if not os.path.exists(valid_dog_path):
    os.mkdir(valid_dog_path)
    for g in range(len(g_valid_dog)):
        # ファイル名から./data/dogs-vs-cats-redux-kernels-edition/train/`の部分を取り除きファイル名にする
        # trainからvalidのディレクトリに画像ファイルを移動
        os.rename(g_valid_dog[g], os.path.join(valid_dog_path, g_valid_dog[g][48:]))
# catディレクトリの処理
if not os.path.exists(valid_cat_path):
    os.mkdir(valid_cat_path)
    for g in range(len(g_valid_cat)):
        # ファイル名から./data/dogs-vs-cats-redux-kernels-edition/train/`の部分を取り除きファイル名にする
        # trainからvalidのディレクトリに画像ファイルを移動
        os.rename(g_valid_cat[g], os.path.join(valid_cat_path, g_valid_cat[g][48:]))

```


```python
# テストデータ
# 全ての.jpgファイルパスのリストを取得
g_test = glob('./data/dogs-vs-cats-redux-kernels-edition/test/*.jpg')

# テストデータにはラベルがないのでunknownディレクトリにデータを格納
test_unknown_path = './data/dogs-vs-cats-redux-kernels-edition/test/unknown'

# unknownディレクトリの処理
if not os.path.exists(test_unknown_path):
    os.mkdir(test_unknown_path)
    for g in range(len(g_test)):
        # ファイル名から./data/dogs-vs-cats-redux-kernels-edition/train/`の部分を取り除きファイル名にする
        # trainからvalidのディレクトリに画像ファイルを移動
        os.rename(g_test[g], os.path.join(test_unknown_path, g_test[g][48:]))

```

## データをテンソルに変換

学習時と評価・テスト時でデータの変換関数を変えます。
- 学習時は汎化性能が上がるように`RandomResizedCrop` や `RandomHorizontalFlip` などデータ拡張する変換を入れる
- 評価時はこのようなランダム性は入れずに入力画像のサイズがネットワークに合うようにサイズを変形する
- `Normalize()` はImageNetの訓練データの平均と分散を表し、入力画像を平均0、分散1に標準化する。ImageNetで学習済みの重みを使うときは必ず入れる変換（2020）


```python
import torch
import torchvision
import torchvision.transforms as transforms

# デバイスモードの取得
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)    # デバイスモードの確認


# transform の定義
# ToTensor で Tensorに変換し
# 画像サイズを(224, 224)で統一
# 標準化 平均0、標準偏差1
# Resizeの前にtransforms.ToTensor()を入れるとエラーになるので注意
train_transform = transforms.Compose(
    [transforms.Resize((224, 224)),                                         # データを(224, 224)に統一する
     transforms.RandomHorizontalFlip(),                                     # ランダムに左右を入れ替える
     transforms.ColorJitter(),                                              # ランダムに画像の閾値を変える
     transforms.RandomRotation(10),                                         # ランダムに画像の回転（±10°）
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 平均0、標準偏差1で標準化
    ]
)

test_transform = transforms.Compose(
    [transforms.Resize((224, 224)),                                         # データを(224, 224)に統一する
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 平均0、標準偏差1で標準化
    ]
)
```

    cpu



```python
# 画像データをデータセットに格納する
# 扱うデータが画像でクラスごとにフォルダ分けされているのでImageFolder()を使用
train_dir = './data/dogs-vs-cats-redux-kernels-edition/train/'
valid_dir = './data/dogs-vs-cats-redux-kernels-edition/valid/'
test_dir = './data/dogs-vs-cats-redux-kernels-edition/test/'

df_train = torchvision.datasets.ImageFolder(
    root = train_dir,
    transform=train_transform
)

df_valid = torchvision.datasets.ImageFolder(
    root = valid_dir,
    transform=test_transform
)

df_test = torchvision.datasets.ImageFolder(
    root = test_dir,
    transform=test_transform
)

# クラスラベルの表示
print(df_train.classes)
print(df_valid.classes)
print(df_test.classes)
```

    ['cat', 'dog']
    ['cat', 'dog']
    ['unknown']


cats がラベル0で dogs がラベル1になっていることがわかります。<br>おそらくアルファベット順にラベルが割り当てられたと<br>unknownはラベル0だけどこれはテスト時にしか使わないので問題なし。

## データローダーの作成

ミニバッチ単位でデータを取得するデータローダーを作成。<br>バッチサイズは128にし、訓練データだけシャッフルします。


```python
from torch.utils.data import DataLoader, TensorDataset    # データ関連のユーティリティクラスのインポート

# 定数（学習方法の設計時）
BATCH_SIZE = 128        # バッチサイズ

# データローダー（loader）の作成 ミニバッチを扱うため、データローダー（loader）を作成する
loader_train = DataLoader(
    df_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)
loader_valid = DataLoader(
    df_valid,
    batch_size=BATCH_SIZE, 
    shuffle=False
)
```

## 作成したデータの確認（1バッチ分データを表示）


```python
import matplotlib.pyplot as plt
%matplotlib inline

def imshow(images, title=None):
    images = images.numpy().transpose((1, 2, 0))  # (h, w, c)
    # denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images + mean
    images = np.clip(images, 0, 1)
    plt.imshow(images)
    if title is not None:
        plt.title(title)

images, classes = next(iter(loader_train))
print(images.size(), classes.size())
grid = torchvision.utils.make_grid(images[:25], nrow=5)
imshow(grid, title='1batch image')
```

    torch.Size([128, 3, 224, 224]) torch.Size([128])



![png](output_20_1.png)


## モデルの定義

#### モデルの読み込み



```python
# 学習済みのモデルをロード
model = torchvision.models.vgg16(pretrained=True).to(device)
model    # モデルの概要の確認
```




    VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace=True)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace=True)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace=True)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace=True)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace=True)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace=True)
        (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): ReLU(inplace=True)
        (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU(inplace=True)
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU(inplace=True)
        (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (25): ReLU(inplace=True)
        (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (27): ReLU(inplace=True)
        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): ReLU(inplace=True)
        (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
      (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace=True)
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace=True)
        (5): Dropout(p=0.5, inplace=False)
        (6): Linear(in_features=4096, out_features=1000, bias=True)
      )
    )



## 出力層の出力を2にする

今回の学習内容は蟻か蜂の2クラス分類なので出力を2に変更します。<br>(classifier) を丸ごと置き換える必要は内容です。


```python
import torch.nn as nn

num_features = model.classifier[6].in_features     # 出力層のニューロン数
OUTPUT_RESULTS = 2        # 出力層のニューロン数

model.classifier[6] = nn.Linear(
    num_features,      # 入力ユニット数
    2                  # 出力結果への出力ユニット数
)

# GPUの使用するかの設定
# use_gpu = torch.cuda.is_available()
# if use_gpu:
#     moedel = model.cuda()
model    # モデル概要の確認
```




    VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace=True)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace=True)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace=True)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace=True)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace=True)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace=True)
        (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): ReLU(inplace=True)
        (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU(inplace=True)
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU(inplace=True)
        (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (25): ReLU(inplace=True)
        (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (27): ReLU(inplace=True)
        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): ReLU(inplace=True)
        (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
      (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace=True)
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace=True)
        (5): Dropout(p=0.5, inplace=False)
        (6): Linear(in_features=4096, out_features=2, bias=True)
      )
    )



## 損失関数の定義<br>オプティマイザー（最適化用オブジェクト）の作成


```python
import torch.optim as optim    # 最適化モジュールのインポート

# 定数
LEARN_RATE = 0.001        # 学習率
MOMENTUM = 0.9            # モーメンタム
# 変数
criterion = nn.CrossEntropyLoss()   # 損失関数：交差エントロピー 学習データの正解率を出力
optimizer = optim.SGD(
    model.parameters(),   # 最適化で更新する重みやバイアスのパラメータ
    lr=LEARN_RATE,        # 学習率
    momentum = MOMENTUM   # モーメンタム
)
```

## 1回分の「訓練（学習）」と「評価」の処理


```python
from torch.autograd import Variable

def train_step(model, criterion, optimizer, train_loader):
    """学習の実行
    訓練モードの設定
    フォワードプロパゲーションで出力結果の取得
    出力結果と正解ラベルから損失および勾配の計算
    勾配を使ってパラメーター（重みとバイアス）の更新
    
    Param:
        model : 定義したモデル
        criterion : 損失関数
        optimizer : オプティマイザー
        train_loader : 訓練データのデータローダー
    """
    # 学習モードの設定
    model.train()
    # 学習中の損失を格納する変数
    running_loss = 0
    # 1ミニバッチ分の「訓練」を実行
    for batch_idx, (images, labels) in enumerate(train_loader):
        # デバイス情報をデータに送る
        images = images.to(device)
        labels = labels.to(device)
        # フォワードプロパゲーションで出力結果を取得
        outputs = model(images)
        
        # 出力結果と正解ラベルから損失を計算し、勾配を計算
        optimizer.zero_grad()    # 勾配を0で初期化

        loss = criterion(outputs, labels)    # 誤差（出力結果と正解ラベルの差）から損失を取得
        running_loss += loss.data[0]         # 損失をpythonの値で追加
        loss.backward()                      # 逆伝播の処理として勾配を計算（自動微分）

        # 勾配を使ってパラメーター（重みとバイアス）を更新
        optimizer.step()                     # 最適化の実施
    # 損失の算出ミニバッチ数分の損失の合計をミニバッチ数で割る
    train_loss = running_loss / len(train_loader)
    return train_loss

def valid_step(model, criterion, valid_loader):
    """評価（推論）検証
    評価モードの設定
    フォワードプロパゲーションで出力結果の取得
    出力結果と正解ラベルから損失の計算
    正解率の算出
    
    Param:
        model : 定義したモデル
        criterion : 損失関数
        optimizer : オプティマイザー
        train_loader : 評価データのデータローダー
    """
    # 評価モードに設定（dropoutなどの挙動が評価用になる）
    model.eval()
    
    running_loss = 0    # 検証中の損失を格納する変数
    correct = 0         # 検証の正解数を格納する変数
    total = 0           # 1ミニバッチ数を格納する変数
    
    with torch.no_grad():                # 勾配は計算しないモードに設定
        # 1ミニバッチ分の「検証」を実行
        for batch_idx, (images, labels) in enumerate(valid_loader):
            # デバイス情報をデータに送る
            images = images.to(device)
            labels = labels.to(device)
            # フォワードプロパゲーションで出力結果を取得
            outputs = model(images)
            
            loss = criterion(outputs, labels)    # 誤差（出力結果と正解ラベルの差）から損失を取得
            running_loss += loss.item()          # 損失をpythonの値で追加
            _, predict = torch.max(outputs, 1)   # 予測した確率の最大値を予測結果として出力
            correct += (predict == labels).sum().item()  # 正解数を取得
            total += labels.size(0)              # 1ミニバッチ数の取得
            
    val_loss = running_loss / len(valid_loader)
    val_acc = float(correct) / total
    
    return val_loss, val_acc
```


```python
log_dir = './data/dogs-vs-cats-redux-kernels-edition/logs/'
# logsディレクトリ
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
```


```python
# 学習にかかる時間を測定する
import time

# 定数
start = time.time()             # 実行開始時間の取得

def init_parameters(layer):
    """パラメータ（重みとバイアス）の初期化
    引数の層が全結合層の時パラメータを初期化する
    
    Param:
      layer: 層情報
    """
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)    # 重みを「一様分布のランダム値」で初期化
        layer.bias.data.fill_(0.0)               # バイアスを「0」で初期化

model.apply(init_parameters)        # 学習の前にパラメーター初期化

EPOCHS = 5    # エポック数
log_dir = './data/dogs-vs-cats-redux-kernels-edition/logs/'

best_acc = 0
loss_list = []
val_loss_list = []
val_acc_list = []
for epoch in range(EPOCHS):
    loss = train_step(model, criterion, optimizer, loader_train)
    val_loss, val_acc = valid_step(model, criterion, loader_valid)

    # 損失や正解率などの情報を表示
    print(f'[Epoch {epoch+1:3d}/{EPOCHS:3d}]' \
          f' loss: {loss:.5f}' \
          f' val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
    
    if val_acc > best_acc:
        print('val_acc improved from %.5f to %.5f!' % (best_acc, val_acc))
        best_acc = val_acc
        model_file = 'epoch%03d-%.3f-%.3f.pth' % (epoch, val_loss, val_acc)
        torch.save(vgg16.state_dict(), os.path.join(log_dir, model_file))

    # logging
    loss_list.append(loss)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    
# 学習終了後、学習に要した時間を出力
print("Computation time:{0:.3f} sec".format(time.time() - start))
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-19-5f12bc8e3a95> in <module>
         26 val_acc_list = []
         27 for epoch in range(EPOCHS):
    ---> 28     loss = train_step(model, criterion, optimizer, loader_train)
         29     val_loss, val_acc = valid_step(model, criterion, loader_valid)
         30 


    <ipython-input-17-9dc5f24b4d1b> in train_step(model, criterion, optimizer, train_loader)
         24         labels = labels.to(device)
         25         # フォワードプロパゲーションで出力結果を取得
    ---> 26         outputs = model(images)
         27 
         28         # 出力結果と正解ラベルから損失を計算し、勾配を計算


    /opt/anaconda3/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        548             result = self._slow_forward(*input, **kwargs)
        549         else:
    --> 550             result = self.forward(*input, **kwargs)
        551         for hook in self._forward_hooks.values():
        552             hook_result = hook(self, input, result)


    /opt/anaconda3/lib/python3.7/site-packages/torchvision/models/vgg.py in forward(self, x)
         41 
         42     def forward(self, x):
    ---> 43         x = self.features(x)
         44         x = self.avgpool(x)
         45         x = torch.flatten(x, 1)


    /opt/anaconda3/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        548             result = self._slow_forward(*input, **kwargs)
        549         else:
    --> 550             result = self.forward(*input, **kwargs)
        551         for hook in self._forward_hooks.values():
        552             hook_result = hook(self, input, result)


    /opt/anaconda3/lib/python3.7/site-packages/torch/nn/modules/container.py in forward(self, input)
         98     def forward(self, input):
         99         for module in self:
    --> 100             input = module(input)
        101         return input
        102 


    /opt/anaconda3/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        548             result = self._slow_forward(*input, **kwargs)
        549         else:
    --> 550             result = self.forward(*input, **kwargs)
        551         for hook in self._forward_hooks.values():
        552             hook_result = hook(self, input, result)


    /opt/anaconda3/lib/python3.7/site-packages/torch/nn/modules/conv.py in forward(self, input)
        347 
        348     def forward(self, input):
    --> 349         return self._conv_forward(input, self.weight)
        350 
        351 class Conv3d(_ConvNd):


    /opt/anaconda3/lib/python3.7/site-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight)
        344                             _pair(0), self.dilation, self.groups)
        345         return F.conv2d(input, weight, self.bias, self.stride,
    --> 346                         self.padding, self.dilation, self.groups)
        347 
        348     def forward(self, input):


    RuntimeError: [enforce fail at CPUAllocator.cpp:64] . DefaultCPUAllocator: can't allocate memory: you tried to allocate 1644167168 bytes. Error code 12 (Cannot allocate memory)



## 損失値の推移グラフ描画



```python
import matplotlib.pyplot as plt
%matplotlib inline

# plot learning curve
plt.figure()
plt.plot(range(EPOCHS), loss_list, 'r-', label='train_loss')
plt.plot(range(EPOCHS), val_loss_list, 'b-', label='val_loss')
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()

plt.figure()
plt.plot(range(EPOCHS), val_acc_list, 'g-', label='val_acc')
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.grid()
```

## テストデータに対する評価


```python
weight_file = './data/dogs-vs-cats-redux-kernels-edition/logs/dogsvscats_ft/epoch003-0.032-0.993.pth'

model = models.vgg16(pretrained=False)
model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(4096, 2)    
)

model.load_state_dict(torch.load(weight_file,
                                 map_location=lambda storage,
                                 loc: storage))
```

テストデータのDataLoaderを作成して最初の128枚の画像をロードする。


```python
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=128,
                                          shuffle=False)

images, _ = iter(test_loader).next()
images = Variable(images, volatile=True)
print(images.size())
```


```python
imshow(torchvision.utils.make_grid(images.data[:25], nrow=5))
```

ネットワークに通して予測しよう


```python
outputs = model(images)
_, predicted = torch.max(outputs.data, 1)
```

predicted には0（ネコ）、1（イヌ）の予測ラベルが入っている


```python
print(predicted.numpy())
```


```python
pred_dogs_images = images[predicted.nonzero().view(-1), :, :, :]
imshow(torchvision.utils.make_grid(pred_dogs_images.data[:25], nrow=5))
```

- predicted.nonzero() で0でない（つまり1）のインデックスが取得できる
- ただし、サイズが (62, 1) のように2D tensorで返ってくるので、view(-1) で1Dtensorにしてからインデキシング

ネコと予測された画像だけ取り出す


```python
pred_cats_images = images[(predicted != 1).nonzero().view(-1), :, :, :]
imshow(torchvision.utils.make_grid(pred_cats_images.data[:25], nrow=5))
```
