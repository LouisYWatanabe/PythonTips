# 事前準備

- フォルダを作成します
- zipファイルをダウンロードします


```python
import os
import urllib.request
import zipfile
```

## フォルダ`data`が存在しない場合に作成する


```python
# 定数
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    # DATA_DIRがなければ
    os.mkdir(DATA_DIR)        # コマンドを実行
```

## ImageNetのclass_indexをダウンロードする

Kerasで提供されているものです
https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py



```python
URL = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
# 保存する場所とファイル名を設定
SAVE_PATH = os.path.join(DATA_DIR, 'imagenet_class_index.json')

if not os.path.exists(SAVE_PATH):
    # SAVE_PATHがなければ
    urllib.request.urlretrieve(URL, SAVE_PATH)    # class_indexをダウンロード
```

## アリとハチの画像データをダウンロードし解凍します

PyTorchのチュートリアルで用意されているものです
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

- zipファイルをダウンロードします
- zipファイルを解凍します
- 解凍後zipファイルを削除します


```python
URL = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
# 保存する場所とファイル名を設定
SAVE_PATH = os.path.join(DATA_DIR, 'hymenoptera_data.zip')

if not os.path.exists(SAVE_PATH):
    # SAVE_PATHがなければ
    urllib.request.urlretrieve(URL, SAVE_PATH)    # zipをダウンロード
    
    # zipファイルを読み込み
    zip = zipfile.ZipFile(SAVE_PATH)
    zip.extractall(DATA_DIR)            # zipファイルを解凍
    zip.close()                         # zipファイルをクローズ
    
    # zipファイルの削除
    os.remove(SAVE_PATH)
```
