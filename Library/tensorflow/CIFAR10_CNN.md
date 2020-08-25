# 画像分類にアンサンブルを使用する

CIFAR-10データセットを同じCNNモデルを複数使用してアンサンブルを行います。


```python
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def prepare_data():
    """
    データの読み込みと前処理
    
    Returns:
        X_train:ndarray
            学習データ(50000.32.32.3)
        X_test(ndarray):
            テストデータ(10000.32.32.3)
        y_train(ndarray):
            学習データのOne-Hot化した正解ラベル(50000,10)
        y_train(ndarray):
            テストデータのOne-Hot化した正解ラベル10000,10)
        y_test_label(ndarray):
            テストデータの正解ラベル(10000)
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # 学習の収束を早めるため標準化
    # 学習データとテスト用データを標準化する
    mean = np.mean(X_train)
    std = np.std(X_train)
    # 標準化する際に分母の標準偏差に極小値を追加し結果が0に限りなく近い値になるのを防ぐ
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    
    # テストデータの正解ラベルを2階テンソルから1階テンソルへフラット化
    y_test_label = y_test.flatten()
    
    # 学習データとテストデータの正解ラベルをOne-Hot表現に変換(10クラス化)
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)
    return X_train, X_test, y_train, y_test, y_test_label 
```


```python
# 画像の読み込みテスト
X_train, X_test, y_train, y_test, y_test_label  = prepare_data()
# データ形状の出力
print('X_train:{}, y_train:{}'.format(X_train.shape, y_train.shape))
print('X_test:{}, y_test:{}'.format(X_test.shape, y_test.shape))
print('y_test_label:{}'.format(y_test_label.shape))
```

    X_train:(50000, 32, 32, 3), y_train:(50000, 10)
    X_test:(10000, 32, 32, 3), y_test:(10000, 10)
    y_test_label:(10000,)
    

## アンサンブルの実装

### CNNを動的に生成する関数の実装

5モデルによるアンサンブルを行うので、動的にモデルを生成する関数を用意します。


```python
from keras.layers import Input, Conv2D, Dense, Activation
from keras.layers import AveragePooling2D, GlobalAvgPool2D
from keras.layers import BatchNormalization
from keras import regularizers
from keras.models import Model

def make_convlayer(input, fsize, layers):
    """
    畳み込み層を生成する

    Parameters: inp(Input): 入力層
                fsize(int): フィルターのサイズ
                layers(int) : 層の数
    Returns:
        Conv2Dを格納したTensorオブジェクト
    """
    x = input
    for i in range(layers):
        x =Conv2D(
            filters=fsize,
            kernel_size=3,
            padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x

def create_model():
    """
    モデルを生成する

    Returns:
      Conv2Dを格納したModelオブジェクト
    """
    input = Input(shape=(32,32,3))
    x = make_convlayer(input, 64, 3)
    x = AveragePooling2D(2)(x)
    x = make_convlayer(x, 128, 3)
    x = AveragePooling2D(2)(x)
    x = make_convlayer(x, 256, 3)
    x = GlobalAvgPool2D()(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(input, x)
    return model
```

## 多数決をとるアンサンブルの実装


```python
from scipy.stats import mode

def ensemble_majority(models, X):
    """
    多数決をとるアンサンブル
    
    Parameters:
        models:list
            Modelオブジェクトリスト
        X:array
            検証用データ
    Returns:
        各画像の正解ラベルを格納した（10000）np.ndarray
    """
    # （データ数、モデル数）のゼロ行列を作成
    pred_labels = np.zeros(
        (
            X.shape[0],   # 行数は画像の枚数と同じ
            len(models)   # 列数はモデルの数
        )
    )
    # modelsからインデックス値と更新をフリーズされたモデルを取り出す
    for i, model in enumerate(models):
        # モデルごとの予測確率(データ数,クラス数)の各行(axis=1)から
        # 最大値のインデックスをとって、(データ数,モデル数)の
        # モデル列の各行にデータの数だけ格納する
        pred_labels[:, i] = np.argmax(model.predict(X), axis=1)
    # mode()でpred_labelsの各行の最頻値のみを[0]指定で取得する
    # (データ数,1)の形状をravel()で(,データ数)の形状にフラット化する    
    return np.ravel(mode(pred_labels, axis=1)[0])
```

## 最高精度を出した時の重みを保存する

学習の終了時にこれまでの精度よりも低い値が出ないように<br>学習中にもっとも高い精度が出た時の重みを保存するようにします。<br>このような処理はエポック終了時に呼ばれる`Callback`クラスの`on_epoch_end`メソッドがあるので<br>これを再定義してエポックごとに重みをファイルを保存する処理を書けばエポックごとに重みを記録することができます。<br>ただ、最高精度が出たときの重みだけを保存したいので、エポックごとに精度をこれまでのものと比較し、より高精度がでた場合にファイルに保存するようにします。




```python
from tensorflow.keras.callbacks import Callback

class Checkpoint(Callback):
    """
    Callbackのサブクラス
    
    Attributes:
        model:object
            学習中のModelオブジェクト
        filepath:str
            重みを保存するファイルのパス
        best_val_acc : 
            最高精度を保持する
    """
    def __init__(self, model, filepath):
        """
        Parameters:
            model:Model
                現在実行中のModelオブジェクト
            filepath:str
                重みを保存するファイルのパス
            best_val_acc(int): 1モデルの最も高い精度を保持
        """
        self.model = model
        self.filepath = filepath
        self.best_val_acc = 0.0

    def on_epoch_end(self, epoch, logs):
        """
        エポック終了時に呼ばれるメソッドをオーバーライド
        
        これまでのエポックより精度が高い場合は重みをファイルに保存する
        
        Parameters:
            epoch(int): エポックの回数
            logs(dict): {'val_acc':損失, 'val_acc':精度}
        """
        if self.best_val_acc < logs['val_acc']:
            # 前回のエポックより精度が高い場合は重みを保存する
            self.model.save_weights(self.filepath)  # ファイルパス
            # 精度をlogsに保存
            self.best_val_acc = logs['val_acc']
            # 重みが保存されたことを精度と共に通知する
            print('Weights saved.', self.best_val_acc)

```


```python
import math
import pickle
import numpy as np

from sklearn.metrics import accuracy_score

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.callbacks import History

def train(X_train, X_test, y_train, y_test, y_test_label):
    """
    学習を行う
    
    Parameters:
        X_train(ndarray): 訓練データ
        X_test(ndarray): 訓練データの正解ラベル
        y_train(ndarray): テストデータ
        y_test(ndarray): テストデータの正解ラベル(One-Hot表)
        y_test_label(ndarray): テストデータの正解ラベル
    """
    models_num  = 5   # アンサンブルするモデルの数
    batch_size = 1024 # ミニバッチの数
    epoch = 80        # エポック数   
    models = []       # モデルを格納するリスト
    # 各モデルの学習履歴を保持するdict
    history_all = {"hists":[], "ensemble_test":[]}
    # 各モデルの推測結果を登録する2階テンソルを0で初期化
    # (データ数, モデル数)
    model_predict = np.zeros((X_test.shape[0], # 行数は画像の枚数
                             models_num))      # 列数はモデルの数

    # モデルの数だけ繰り返す
    for i in range(models_num):
        # 何番目のモデルかを表示
        print('Model',i+1)
        # CNNのモデルを生成
        train_model = create_model()
        # モデルをコンパイルする
        train_model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=["acc"])
        # コンパイル後のモデルをリストに追加
        models.append(train_model)

        # コールバックに登録するHistoryオブジェクトを生成
        hist = History()
        # コールバックに登録するCheckpointオブジェクトを生成
        cpont = Checkpoint(train_model,       # Modelオブジェクト
                           f'weights_{i}.h5') # 重みを保存するファイル名
        # ステップ減衰関数
        def step_decay(epoch):
            initial_lrate = 0.001 # ベースにする学習率
            drop = 0.5            # 減衰率
            epochs_drop = 10.0    # ステップ減衰は10エポックごと
            lrate = initial_lrate * math.pow(
                drop,
                math.floor((1+epoch)/epochs_drop)
            )
            return lrate
            
        lrate = LearningRateScheduler(step_decay) # スケジューラ―オブジェクト
        
        # データ拡張
        datagen = ImageDataGenerator(
            rotation_range=15,      # 15度の範囲でランダムに回転させる
            width_shift_range=0.1,  # 横サイズの0.1の割合でランダムに水平移動
            height_shift_range=0.1, # 縦サイズの0.1の割合でランダムに垂直移動
            horizontal_flip=True,   # 水平方向にランダムに反転、左右の入れ替え
            zoom_range=0.2,         # ランダムに拡大
            )

        # 学習を行う
        train_model.fit_generator(
            datagen.flow(X_train,
                         y_train,
                         batch_size=batch_size),
            epochs=epoch,
            steps_per_epoch=X_train.shape[0] // batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[hist, cpont, lrate] # コールバック
            )       

        # 学習に用いたモデルで最も精度が高かったときの重みを読み込む
        train_model.load_weights(f'weights_{i}.h5')
        
        # 対象のモデルのすべての重み更新をフリーズする
        for layer in train_model.layers:
            layer.trainable = False

        # テストデータで推測し、各画像ごとにラベルの最大値を求め、
        # 対象のインデックスを正解ラベルとしてmodel_predictのi列に格納
        model_predict[:, i] = np.argmax(train_model.predict(X_test),
                                       axis=-1) # 行ごとの最大値を求める

        # 学習に用いたモデルの学習履歴をhistory_allのhistsキーに登録
        history_all['hists'].append(hist.history)
        
        # 多数決のアンサンブルを実行
        ensemble_test_pred = ensemble_majority(models, X_test)
        
        # scikit-learn.accuracy_score()でアンサンブルによる精度を取得
        ensemble_test_acc = accuracy_score(y_test_label, ensemble_test_pred)
        
        # アンサンブルの精度をhistory_allのensemble_testキーに追加
        history_all['ensemble_test'].append(ensemble_test_acc)
        # 現在のアンサンブルの精度を出力
        print('Current Ensemble Accuracy : ', ensemble_test_acc)

    history_all['corrcoef'] = np.corrcoef(model_predict,
                                          rowvar=False) # 列ごとの相関係数を求める
    print('Correlation predicted value')
    print(history_all['corrcoef'])
```


```python
# 実行部

# データを用意する
X_train, X_test, y_train, y_test, y_test_label  = prepare_data()

# アンサンブルを実行
train(X_train, X_test, y_train, y_test, y_test_label)
```

    Model 1
    WARNING:tensorflow:From <ipython-input-6-90ca68dba4c1>:74: Model.fit_generator (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use Model.fit, which supports generators.
    Epoch 1/80
    48/48 [==============================] - ETA: 0s - loss: 1.6129 - acc: 0.4102 Weights saved. 0.15520000457763672
    48/48 [==============================] - 1284s 27s/step - loss: 1.6129 - acc: 0.4102 - val_loss: 2.4185 - val_acc: 0.1552
    Epoch 2/80
    48/48 [==============================] - ETA: 0s - loss: 1.1929 - acc: 0.5722 Weights saved. 0.1737000048160553
    48/48 [==============================] - 1714s 36s/step - loss: 1.1929 - acc: 0.5722 - val_loss: 2.9431 - val_acc: 0.1737
    Epoch 3/80
    48/48 [==============================] - 1188s 25s/step - loss: 0.9764 - acc: 0.6519 - val_loss: 3.0561 - val_acc: 0.1651
    Epoch 4/80
    48/48 [==============================] - ETA: 0s - loss: 0.8435 - acc: 0.7038 Weights saved. 0.24969999492168427
    48/48 [==============================] - 1203s 25s/step - loss: 0.8435 - acc: 0.7038 - val_loss: 2.3890 - val_acc: 0.2497
    Epoch 5/80
    48/48 [==============================] - ETA: 0s - loss: 0.7357 - acc: 0.7428 Weights saved. 0.31040000915527344
    48/48 [==============================] - 1191s 25s/step - loss: 0.7357 - acc: 0.7428 - val_loss: 2.1275 - val_acc: 0.3104
    Epoch 6/80
    48/48 [==============================] - ETA: 0s - loss: 0.6508 - acc: 0.7737 Weights saved. 0.5045999884605408
    48/48 [==============================] - 1132s 24s/step - loss: 0.6508 - acc: 0.7737 - val_loss: 1.4007 - val_acc: 0.5046
    Epoch 7/80
    48/48 [==============================] - ETA: 0s - loss: 0.6034 - acc: 0.7884 Weights saved. 0.6292999982833862
    48/48 [==============================] - 1288s 27s/step - loss: 0.6034 - acc: 0.7884 - val_loss: 1.1437 - val_acc: 0.6293
    Epoch 8/80
    48/48 [==============================] - ETA: 0s - loss: 0.5595 - acc: 0.8056 Weights saved. 0.7184000015258789
    48/48 [==============================] - 1154s 24s/step - loss: 0.5595 - acc: 0.8056 - val_loss: 0.8273 - val_acc: 0.7184
    Epoch 9/80
    48/48 [==============================] - ETA: 0s - loss: 0.5124 - acc: 0.8229 Weights saved. 0.7472000122070312
    48/48 [==============================] - 1125s 23s/step - loss: 0.5124 - acc: 0.8229 - val_loss: 0.7268 - val_acc: 0.7472
    Epoch 10/80
    48/48 [==============================] - ETA: 0s - loss: 0.4403 - acc: 0.8477 Weights saved. 0.7763000130653381
    48/48 [==============================] - 1134s 24s/step - loss: 0.4403 - acc: 0.8477 - val_loss: 0.6388 - val_acc: 0.7763
    Epoch 11/80
    48/48 [==============================] - ETA: 0s - loss: 0.4151 - acc: 0.8564 Weights saved. 0.8166000247001648
    48/48 [==============================] - 1341s 28s/step - loss: 0.4151 - acc: 0.8564 - val_loss: 0.5398 - val_acc: 0.8166
    Epoch 12/80
    48/48 [==============================] - ETA: 0s - loss: 0.3881 - acc: 0.8651 Weights saved. 0.8485999703407288
    48/48 [==============================] - 1548s 32s/step - loss: 0.3881 - acc: 0.8651 - val_loss: 0.4494 - val_acc: 0.8486
    Epoch 13/80
    48/48 [==============================] - 1113s 23s/step - loss: 0.3732 - acc: 0.8710 - val_loss: 0.5132 - val_acc: 0.8282
    Epoch 14/80
    48/48 [==============================] - ETA: 0s - loss: 0.3623 - acc: 0.8724 Weights saved. 0.8629000186920166
    48/48 [==============================] - 1085s 23s/step - loss: 0.3623 - acc: 0.8724 - val_loss: 0.4029 - val_acc: 0.8629
    Epoch 15/80
    48/48 [==============================] - 1096s 23s/step - loss: 0.3502 - acc: 0.8781 - val_loss: 0.5199 - val_acc: 0.8267
    Epoch 16/80
    48/48 [==============================] - 1059s 22s/step - loss: 0.3398 - acc: 0.8802 - val_loss: 0.6218 - val_acc: 0.7930
    Epoch 17/80
    48/48 [==============================] - 1037s 22s/step - loss: 0.3249 - acc: 0.8874 - val_loss: 0.5306 - val_acc: 0.8200
    Epoch 18/80
    48/48 [==============================] - 1043s 22s/step - loss: 0.3107 - acc: 0.8922 - val_loss: 0.4678 - val_acc: 0.8484
    Epoch 19/80
    48/48 [==============================] - 1046s 22s/step - loss: 0.3035 - acc: 0.8936 - val_loss: 0.4802 - val_acc: 0.8392
    Epoch 20/80
    48/48 [==============================] - 1040s 22s/step - loss: 0.2584 - acc: 0.9101 - val_loss: 0.4344 - val_acc: 0.8530
    Epoch 21/80
    48/48 [==============================] - ETA: 0s - loss: 0.2443 - acc: 0.9168 Weights saved. 0.8751999735832214
    48/48 [==============================] - 1040s 22s/step - loss: 0.2443 - acc: 0.9168 - val_loss: 0.3668 - val_acc: 0.8752
    Epoch 22/80
    48/48 [==============================] - 1050s 22s/step - loss: 0.2342 - acc: 0.9187 - val_loss: 0.3974 - val_acc: 0.8669
    Epoch 23/80
    48/48 [==============================] - 1043s 22s/step - loss: 0.2276 - acc: 0.9218 - val_loss: 0.3986 - val_acc: 0.8674
    Epoch 24/80
    48/48 [==============================] - 1059s 22s/step - loss: 0.2209 - acc: 0.9246 - val_loss: 0.3984 - val_acc: 0.8663
    Epoch 25/80
    48/48 [==============================] - 1043s 22s/step - loss: 0.2205 - acc: 0.9244 - val_loss: 0.5206 - val_acc: 0.8403
    Epoch 26/80
    48/48 [==============================] - ETA: 0s - loss: 0.2113 - acc: 0.9268 Weights saved. 0.8809000253677368
    48/48 [==============================] - 1048s 22s/step - loss: 0.2113 - acc: 0.9268 - val_loss: 0.3632 - val_acc: 0.8809
    Epoch 27/80
    48/48 [==============================] - 1043s 22s/step - loss: 0.2058 - acc: 0.9289 - val_loss: 0.4156 - val_acc: 0.8681
    Epoch 28/80
    48/48 [==============================] - 1041s 22s/step - loss: 0.2026 - acc: 0.9306 - val_loss: 0.3846 - val_acc: 0.8739
    Epoch 29/80
    48/48 [==============================] - 1046s 22s/step - loss: 0.1967 - acc: 0.9328 - val_loss: 0.4209 - val_acc: 0.8657
    Epoch 30/80
    48/48 [==============================] - ETA: 0s - loss: 0.1724 - acc: 0.9412 Weights saved. 0.8931000232696533
    48/48 [==============================] - 1045s 22s/step - loss: 0.1724 - acc: 0.9412 - val_loss: 0.3247 - val_acc: 0.8931
    Epoch 31/80
    48/48 [==============================] - 1045s 22s/step - loss: 0.1651 - acc: 0.9442 - val_loss: 0.3511 - val_acc: 0.8881
    Epoch 32/80
    48/48 [==============================] - 1045s 22s/step - loss: 0.1585 - acc: 0.9466 - val_loss: 0.4085 - val_acc: 0.8744
    Epoch 33/80
    48/48 [==============================] - 1185s 25s/step - loss: 0.1581 - acc: 0.9464 - val_loss: 0.3719 - val_acc: 0.8817
    Epoch 34/80
    48/48 [==============================] - 1703s 35s/step - loss: 0.1578 - acc: 0.9466 - val_loss: 0.3504 - val_acc: 0.8875
    Epoch 35/80
    11/48 [=====>........................] - ETA: 15:47 - loss: 0.1476 - acc: 0.9517


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-7-304c36019ee4> in <module>
          5 
          6 # アンサンブルを実行
    ----> 7 train(X_train, X_test, y_train, y_test, y_test_label)
    

    <ipython-input-6-90ca68dba4c1> in train(X_train, X_test, y_train, y_test, y_test_label)
         72 
         73         # 学習を行う
    ---> 74         train_model.fit_generator(
         75             datagen.flow(X_train,
         76                          y_train,
    

    ~\anaconda3\lib\site-packages\tensorflow\python\util\deprecation.py in new_func(*args, **kwargs)
        322               'in a future version' if date is None else ('after %s' % date),
        323               instructions)
    --> 324       return func(*args, **kwargs)
        325     return tf_decorator.make_decorator(
        326         func, new_func, 'deprecated',
    

    ~\anaconda3\lib\site-packages\tensorflow\python\keras\engine\training.py in fit_generator(self, generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, validation_freq, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)
       1813     """
       1814     _keras_api_gauge.get_cell('fit_generator').set(True)
    -> 1815     return self.fit(
       1816         generator,
       1817         steps_per_epoch=steps_per_epoch,
    

    ~\anaconda3\lib\site-packages\tensorflow\python\keras\engine\training.py in _method_wrapper(self, *args, **kwargs)
        106   def _method_wrapper(self, *args, **kwargs):
        107     if not self._in_multi_worker_mode():  # pylint: disable=protected-access
    --> 108       return method(self, *args, **kwargs)
        109 
        110     # Running inside `run_distribute_coordinator` already.
    

    ~\anaconda3\lib\site-packages\tensorflow\python\keras\engine\training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
       1096                 batch_size=batch_size):
       1097               callbacks.on_train_batch_begin(step)
    -> 1098               tmp_logs = train_function(iterator)
       1099               if data_handler.should_sync:
       1100                 context.async_wait()
    

    ~\anaconda3\lib\site-packages\tensorflow\python\eager\def_function.py in __call__(self, *args, **kwds)
        778       else:
        779         compiler = "nonXla"
    --> 780         result = self._call(*args, **kwds)
        781 
        782       new_tracing_count = self._get_tracing_count()
    

    ~\anaconda3\lib\site-packages\tensorflow\python\eager\def_function.py in _call(self, *args, **kwds)
        805       # In this case we have created variables on the first call, so we run the
        806       # defunned version which is guaranteed to never create variables.
    --> 807       return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
        808     elif self._stateful_fn is not None:
        809       # Release the lock early so that multiple threads can perform the call
    

    ~\anaconda3\lib\site-packages\tensorflow\python\eager\function.py in __call__(self, *args, **kwargs)
       2827     with self._lock:
       2828       graph_function, args, kwargs = self._maybe_define_function(args, kwargs)
    -> 2829     return graph_function._filtered_call(args, kwargs)  # pylint: disable=protected-access
       2830 
       2831   @property
    

    ~\anaconda3\lib\site-packages\tensorflow\python\eager\function.py in _filtered_call(self, args, kwargs, cancellation_manager)
       1841       `args` and `kwargs`.
       1842     """
    -> 1843     return self._call_flat(
       1844         [t for t in nest.flatten((args, kwargs), expand_composites=True)
       1845          if isinstance(t, (ops.Tensor,
    

    ~\anaconda3\lib\site-packages\tensorflow\python\eager\function.py in _call_flat(self, args, captured_inputs, cancellation_manager)
       1921         and executing_eagerly):
       1922       # No tape is watching; skip to running the function.
    -> 1923       return self._build_call_outputs(self._inference_function.call(
       1924           ctx, args, cancellation_manager=cancellation_manager))
       1925     forward_backward = self._select_forward_and_backward_functions(
    

    ~\anaconda3\lib\site-packages\tensorflow\python\eager\function.py in call(self, ctx, args, cancellation_manager)
        543       with _InterpolateFunctionError(self):
        544         if cancellation_manager is None:
    --> 545           outputs = execute.execute(
        546               str(self.signature.name),
        547               num_outputs=self._num_outputs,
    

    ~\anaconda3\lib\site-packages\tensorflow\python\eager\execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
         57   try:
         58     ctx.ensure_initialized()
    ---> 59     tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
         60                                         inputs, attrs, num_outputs)
         61   except core._NotOkStatusException as e:
    

    KeyboardInterrupt: 



```python

```
