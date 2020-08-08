# データフレームのメモリ削減関数

---
title: めっちゃ使えるpandasのメモリサイズをグッと抑える汎用的な関数
tags: Python pandas
author: hiroyuki_kageyama
slide: false
---
# はじめに
kaggleでpandasを使用していると、とんでもなく重いデータを扱わなければいけないことがあります。
どうすればいいのか途方に暮れていたところ、kaggleのカーネルに汎用的で使い勝手のいいコードが投稿されていたので紹介します。
※以下のコードはFabienDaniel氏による投稿を引用しています。偉大な先駆者に感謝です。
引用元：https://www.kaggle.com/fabiendaniel/elo-world

## データフレームのメモリ削減関数
```python:script.py
import pandas as pd
import numpy as np

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns: #columns毎に処理
        col_type = df[col].dtypes
        if col_type in numerics: #numericsのデータ型の範囲内のときに処理を実行. データの最大最小値を元にデータ型を効率的なものに変更
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
```

### 解説
pandasに詳しい方はご存知と思いますが、
csvから読み込んだデータなどは、データ型を指定しなければ`Int64`,`float64`が指定されてしまいます。
大量の数値データを扱う場合は、この指定がネックとなりメモリを大量に使用してしまうはめに。
上記コードは、カラムのデータ型がintかfloatの場合に最大値・最小値を鑑みて、適切なデータ型を指定するようにできています。

#### 使用例
```python:script.py
print('Importing data...')
historical_transactions = pd.read_csv('historical_transactions.csv')
print('start size: {:5.2f} Mb'.format(historical_transactions.memory_usage().sum() / 1024**2))
historical_transactions = reduce_mem_usage(historical_transactions)
```

<strong>【実行結果】</strong>
Importing data...
start size: <strong>3109.54 Mb</strong>
Mem. usage decreased to <strong>1749.11 Mb (43.7% reduction)</strong>

使用例に用いたcsvは読み込んだ時に3109.54 Mbありましたが、
無事1749.11 Mbまで軽くすることができました。
