# MNISTをCNNで分類する

kaggleの[Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/overview)（MNISTデータセット）を使用して、`28×28`のグレースケール画像データに対して`0~9`のラベルを予測します。<br>予測はKerasを用いたCNNで行います。


```python
!ls ./data
```

    digit-recognizer
    house-prices-advanced-regression-techniques
    


```python
!ls ./data/digit-recognizer/
```

    sample_submission.csv
    test.csv
    train.csv
    


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
path = './data/digit-recognizer/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 785 columns</p>
</div>




```python
train.shape
```




    (42000, 785)



## 前処理


```python
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
```


```python
# trainデータから画像データを抽出
train_x = train.drop(['label'], axis=1)
# trainデータから正解ラベルを抽出
train_y = train['label']
```


```python
# trainデータを4分割して学習：検証=3:1とする
kf = KFold(n_splits=4, shuffle=True, random_state=42)
# 学習用と検証用のレコードのインデックス配列を取得
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_idx[:5]
```




    array([0, 2, 3, 5, 8])




```python
va_idx[:5]
```




    array([ 1,  4,  6,  7, 13])




```python
# 学習用と検証用の画像データと正解ラベルをそれぞれ取得
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
```


```python
# 画像のピクセルを255で割り、0~1の範囲にしてNumpy.arrayにする
tr_x, va_x = np.array(tr_x / 255.0), np.array(va_x / 255.0)
```

## データ形状の変換

説明変数である、`tr_x, va_x`を`(データ数, 28, 28, 1)`の形状に変更


```python
# 画像データの形状の変更
tr_x, va_x = tr_x.reshape(-1, 28, 28, 1), va_x.reshape(-1, 28, 28, 1)
```


```python
print('before tr_y {}'.format(tr_y))
# 正解ラベルをone-Hot表現する
tr_y = to_categorical(tr_y, 10)
va_y = to_categorical(va_y, 10)

print('after tr_y {}'.format(tr_y))
```

    before tr_y 0        1
    2        1
    3        4
    5        0
    8        5
            ..
    41995    0
    41996    1
    41997    7
    41998    6
    41999    9
    Name: label, Length: 31500, dtype: int64
    after tr_y [[0. 1. 0. ... 0. 0. 0.]
     [0. 1. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 1. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 1.]]
    


```python
# tr_x, va_x, tr_y, va_yの形状の表示
print(tr_x.shape)
print(va_x.shape)
print(tr_y.shape)
print(va_y.shape)
```

    (31500, 28, 28, 1)
    (10500, 28, 28, 1)
    (31500, 10)
    (10500, 10)
    

学習データと検証データともに、2次元のNumpy配列に格納されていることが確認できます。<br>1次元が画像の枚数、2次元が画像のデータ数です。

学習データに格納された正解ラベルの分布を表示します。


```python
from collections import Counter
# 格納された各数字の枚数をカウント
count = Counter(train['label'])
count
```




    Counter({1: 4684,
             0: 4132,
             4: 4072,
             7: 4401,
             3: 4351,
             5: 3795,
             8: 4063,
             9: 4188,
             2: 4177,
             6: 4137})




```python
# 各数字の枚数をカウント
import seaborn as sns
# 0~9までの数字の枚数をグラフ化する
sns.countplot(train['label'])
# 表示スタイルの変更
sns.set(context='talk')
```


![png](output_21_0.png)



```python
# 画像1枚分のデータを出力
# print(tr_x[0])
```

## 画像データの視覚化

学習データの1~50枚目までを描画します。


```python
# 学習データの描画
plt.figure(figsize=(12, 10))

for i in range(50):
    # 5行10列の画像表示場所の設定
    plt.subplot(5, 10, i+1)
    # グレースケール
    plt.gray()
    # 28×28にリサイズする
    plt.imshow(tr_x[i].reshape((28, 28)), interpolation='nearest')
    
plt.show()
```


![png](output_24_0.png)


## CNNモデルの構築

1層構造のCNNモデルを構築します。<br>Kerasの畳み込み層を生成する`Conv2D()`メソッドは

> データサイズ、行データ、列データ、チャネル

という形状の4階テンソルを入力として受け取るようになっています。<br>チャネルは画像のピクセル値を格納するための次元で、カラー画像に対応できるように用意されたものです。<br>グレースケール画像の場合は輝度値を表す1、カラー画像の場合は、RGBの3を指定します。

ここでは、CNNモデルに以下の内容を追加してモデルを構築します。

- プーリング層の追加
- ドロップアウトの追加
- 畳み込み層を複数配置
- Flatten層以降に全結合層の追加

畳み込み層、畳み込み層、プーリング層、畳み込み層、プーリング層、全結合層、出力層の流れで構築を行います。


```python
tr_x.shape, va_x.shape
```




    ((31500, 28, 28, 1), (10500, 28, 28, 1))



Sequentialモデルを`.add`で作成します。


```python
# モデル構築
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Sequentialオブジェクトの生成
model = Sequential()

# 第1層：畳み込み層
model.add(
    Conv2D(
        filters=32,                   # フィルター数
        kernel_size=(5, 5),           # 5×5のフィルター 32×25＝800個の重みと各フィルターに0で初期化されたバイアスが1つずつ32こ用意されます
        padding='same',               # ゼロパディング
        input_shape=(28, 28, 1),      # 入力データの形状
        activation='relu',            # 活性化関数はRelu
        name='Conv_1'                 # 表示用の名称
    )
)
# 第2層：畳み込み層
model.add(
    Conv2D(
        filters=64,                   # フィルター数
        kernel_size=(7, 7),           # パラメータ数=前層のフィルター32*(7*7*64)+バイアス64= 100416
        padding='same',               # ゼロパディング
        activation='relu',            # 活性化関数はRelu
        name='Conv_2'                 # 表示用の名称
    )
)
# 第3層：プーリング層 14
model.add(
    MaxPooling2D(
        pool_size=(2, 2)              # ウィンドウサイズ(2*2)
    )
)

# ドロップアウト
model.add(Dropout(0.5))

# 第4層：畳み込み層
model.add(
    Conv2D(
        filters=64,                   # フィルター数
        kernel_size=(5, 5),           # パラメータ数=前層のフィルター64*(5*5*64)+バイアス64= 102464
        padding='same',               # ゼロパディング
        activation='relu',            # 活性化関数はRelu
        name='Conv_4'                 # 表示用の名称
    )
)
# 第5層：畳み込み層
model.add(
    Conv2D(
        filters=32,                   # フィルター数
        kernel_size=(3, 3),           # パラメータ数=前層のフィルター64*(3*3*32)+バイアス32= 18464
        padding='same',               # ゼロパディング
        activation='relu',            # 活性化関数はRelu
        name='Conv_5'                 # 表示用の名称
    )
)
# 第6層：プーリング層 7
model.add(
    MaxPooling2D(
        pool_size=(2, 2)              # ウィンドウサイズ(2*2)
    )
)
# ドロップアウト
model.add(Dropout(0.55))

# Flatten層
model.add(Flatten())                  # ユニット数7*7*32=1568

# 第7層：全結合層
model.add(
    Dense(
        700,                          # 出力ニューロン数 700 パラメータ数は 700*1568=1097600
        activation='relu',            # 活性化関数はRelu
        name='layer_7'                # 表示用の名称
    )

)
model.add(Dropout(0.3))

# 第8層：全結合層
model.add(
    Dense(
        150,                          # 出力ニューロン数 150 パラメータ数は 150*700=105000
        activation='relu',            # 活性化関数はRelu
        name='layer_8'                # 表示用の名称
    )
)
model.add(Dropout(0.35))
# 出力層
model.add(
    Dense(
        10,                           # 出力ニューロン数 10
        activation='softmax',         # マルチクラス分類用の活性化関数を指定
        name='layer_out'              # 表示用の名称
    )
)

model.compile(
    # 損失関数をクロスエントロピー誤差
    loss='categorical_crossentropy',
    # オプティマイザーはAdamを指定
    optimizer='rmsprop',
    # 学習評価として正解率を指定
    metrics=['accuracy']
)
# モデルの構造を出力
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    Conv_1 (Conv2D)              (None, 28, 28, 32)        832       
    _________________________________________________________________
    Conv_2 (Conv2D)              (None, 28, 28, 64)        100416    
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0         
    _________________________________________________________________
    dropout (Dropout)            (None, 14, 14, 64)        0         
    _________________________________________________________________
    Conv_4 (Conv2D)              (None, 14, 14, 64)        102464    
    _________________________________________________________________
    Conv_5 (Conv2D)              (None, 14, 14, 32)        18464     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 32)          0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 7, 7, 32)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 1568)              0         
    _________________________________________________________________
    layer_7 (Dense)              (None, 700)               1098300   
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 700)               0         
    _________________________________________________________________
    layer_8 (Dense)              (None, 150)               105150    
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 150)               0         
    _________________________________________________________________
    layer_out (Dense)            (None, 10)                1510      
    =================================================================
    Total params: 1,427,136
    Trainable params: 1,427,136
    Non-trainable params: 0
    _________________________________________________________________
    

## 学習の実行

学習回数は20回、ミニバッチ数は100で学習を行います。<br>データ拡張を4つの処理で実行します。

- 左右への移動
- 上下への移動
- 回転
- 拡大


```python
# 学習にかかる時間を測定する
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# データ拡張
datagen = ImageDataGenerator(
    width_shift_range=0.1,  # 横サイズの0.1の割合でランダムに水平移動
    height_shift_range=0.1, # 縦サイズの0.1の割合でランダムに垂直移動
    rotation_range=10,      # 10度の範囲でランダムに回転させる
    zoom_range=0.1          # ランダムに拡大
    )
# ミニバッチのサイズ
batch_size = 100
# 学習回数
epochs = 20

# 定数
start = time.time()         # 実行開始時間の取得

# 学習の実行
hist = model.fit(
    datagen.flow(
        tr_x,
        tr_y,
        batch_size,
        shuffle=False
    ),
    # 1回の学習におけるステップ数
    # 画像の枚数をミニバッチのサイズで割った整数値
    steps_per_epoch=tr_x.shape[0] // batch_size,
    epochs=epochs,          # 学習回数
    verbose=1,              # 学習の進捗状況を出力する
    # テストデータ
    validation_data=(va_x,va_y)
)
print('Finished Training')
# 学習終了後、学習に要した時間を出力
print('Computation time:{0:.3f} sec'.format(time.time() - start))
```

    Epoch 1/20
    315/315 [==============================] - 205s 649ms/step - loss: 0.5761 - accuracy: 0.8123 - val_loss: 0.0744 - val_accuracy: 0.9796
    Epoch 2/20
    315/315 [==============================] - 203s 645ms/step - loss: 0.1557 - accuracy: 0.9544 - val_loss: 0.0378 - val_accuracy: 0.9884
    Epoch 3/20
    315/315 [==============================] - 206s 653ms/step - loss: 0.1140 - accuracy: 0.9662 - val_loss: 0.0434 - val_accuracy: 0.9877
    Epoch 4/20
    315/315 [==============================] - 299s 950ms/step - loss: 0.0989 - accuracy: 0.9719 - val_loss: 0.0336 - val_accuracy: 0.9898
    Epoch 5/20
    315/315 [==============================] - 218s 691ms/step - loss: 0.0899 - accuracy: 0.9746 - val_loss: 0.0421 - val_accuracy: 0.9892
    Epoch 6/20
    315/315 [==============================] - 202s 641ms/step - loss: 0.0863 - accuracy: 0.9759 - val_loss: 0.0328 - val_accuracy: 0.9914
    Epoch 7/20
    315/315 [==============================] - 203s 644ms/step - loss: 0.0801 - accuracy: 0.9783 - val_loss: 0.0311 - val_accuracy: 0.9921
    Epoch 8/20
    315/315 [==============================] - 200s 634ms/step - loss: 0.0753 - accuracy: 0.9803 - val_loss: 0.0380 - val_accuracy: 0.9908
    Epoch 9/20
    315/315 [==============================] - 195s 620ms/step - loss: 0.0761 - accuracy: 0.9787 - val_loss: 0.0363 - val_accuracy: 0.9919
    Epoch 10/20
    315/315 [==============================] - 205s 652ms/step - loss: 0.0761 - accuracy: 0.9813 - val_loss: 0.0313 - val_accuracy: 0.9916
    Epoch 11/20
    315/315 [==============================] - 193s 613ms/step - loss: 0.0746 - accuracy: 0.9800 - val_loss: 0.0314 - val_accuracy: 0.9931
    Epoch 12/20
    315/315 [==============================] - 195s 618ms/step - loss: 0.0776 - accuracy: 0.9803 - val_loss: 0.0314 - val_accuracy: 0.9926
    Epoch 13/20
    315/315 [==============================] - 197s 627ms/step - loss: 0.0768 - accuracy: 0.9795 - val_loss: 0.0305 - val_accuracy: 0.9923
    Epoch 14/20
    315/315 [==============================] - 194s 617ms/step - loss: 0.0755 - accuracy: 0.9804 - val_loss: 0.0334 - val_accuracy: 0.9925
    Epoch 15/20
    315/315 [==============================] - 193s 612ms/step - loss: 0.0792 - accuracy: 0.9811 - val_loss: 0.0450 - val_accuracy: 0.9909
    Epoch 16/20
    315/315 [==============================] - 191s 607ms/step - loss: 0.0778 - accuracy: 0.9810 - val_loss: 0.0411 - val_accuracy: 0.9918
    Epoch 17/20
    315/315 [==============================] - 204s 647ms/step - loss: 0.0836 - accuracy: 0.9803 - val_loss: 0.0330 - val_accuracy: 0.9925
    Epoch 18/20
    315/315 [==============================] - 196s 624ms/step - loss: 0.0797 - accuracy: 0.9800 - val_loss: 0.0441 - val_accuracy: 0.9926
    Epoch 19/20
    315/315 [==============================] - 194s 616ms/step - loss: 0.0803 - accuracy: 0.9810 - val_loss: 0.0362 - val_accuracy: 0.9932
    Epoch 20/20
    315/315 [==============================] - 203s 645ms/step - loss: 0.0820 - accuracy: 0.9807 - val_loss: 0.0608 - val_accuracy: 0.9918
    Finished Training
    Computation time:4117.517 sec
    

## 損失と正解率の推移をグラフ化


```python
# プロットサイズの設定
plt.figure(figsize=(15, 6))
# プロット図を縮小して図の間のスペースを確保
plt.subplots_adjust(wspace=0.2)
# 2つの図を並べて表示
plt.subplot(1, 2, 1)
# 学習データのlossの表示
plt.plot(
    hist.history['loss'],
    label='training',
    color='black'
)
# 検証データのlossの表示
plt.plot(
    hist.history['val_loss'],
    label='test',
    color='red'
)

plt.ylim((0, 1))
plt.legend()        # 凡例
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')

plt.subplot(1, 2, 2)
# 学習データの正解率の表示
plt.plot(
    hist.history['accuracy'],
    label='training',
    color='black'
)
# 検証データの正解率の表示
plt.plot(
    hist.history['val_accuracy'],
    label='test',
    color='red'
)

plt.ylim((0.5, 1))
plt.legend()        # 凡例
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
```




    Text(0, 0.5, 'accuracy')




![png](output_32_1.png)


# テストデータで予測を推論を実施しcsvデータの作成


```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 784 columns</p>
</div>




```python
# 画像のピクセル値を255.0で割って0～1.0の範囲にしてnumpy.arrayに変換。
test = np.array(test / 255.0)
```


```python
test.shape
```




    (28000, 784)




```python
# 画像データの形状の変更
test = test.reshape(-1, 28, 28, 1)
```


```python
# テストデータで予測を実施しNumpy配列に代入
result = model.predict(test)
# 予測結果の先頭から5番目までを出力
result[:5]
```




    array([[9.0328206e-19, 1.3224123e-14, 1.0000000e+00, 2.5455612e-11,
            4.9241468e-14, 2.5259075e-17, 1.5714953e-20, 1.1636398e-10,
            5.2612618e-13, 7.8641995e-15],
           [9.9999702e-01, 5.7220905e-11, 3.3378303e-08, 3.3752934e-09,
            1.4413253e-08, 1.6902433e-08, 1.4004927e-06, 1.1326386e-09,
            1.3621002e-07, 1.4256221e-06],
           [6.7240388e-38, 0.0000000e+00, 6.5172960e-35, 2.9886490e-31,
            7.0922987e-18, 1.3095104e-30, 0.0000000e+00, 1.9565715e-27,
            3.6929768e-26, 1.0000000e+00],
           [9.9990320e-01, 2.5112037e-10, 7.3365464e-07, 1.7500643e-08,
            2.1111265e-07, 2.8331947e-08, 1.9334202e-06, 3.1483228e-08,
            5.4624411e-07, 9.3386094e-05],
           [4.1663351e-29, 4.9404130e-25, 3.1350249e-19, 1.0000000e+00,
            1.9453316e-29, 3.0350502e-18, 2.0458310e-32, 8.5880880e-21,
            1.6438069e-19, 5.4676083e-22]], dtype=float32)




```python
# 最大のインデックスを予測した数値として出力
print([x.argmax() for x in result[:5]])
# 予測した数値をNumpy配列に代入
y_test = [x.argmax() for x in result]
```

    [2, 0, 9, 0, 3]
    


```python
# 予測結果を先頭から5つ表示
y_test[:5]
```




    [2, 0, 9, 0, 3]




```python
submit_df = pd.read_csv(path+'sample_submission.csv')
submit_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ImageId</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Label行に予測値を格納する
submit_df['Label'] = y_test
submit_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ImageId</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



## 予測データファイルをCSVに保存


```python
# CSVファイルに保存
submit_df.to_csv('submission_cnn_dataaug.csv', index=False)
```
