# csvの読み込み

```python
import pandas as pd
# csvの読み込み
pd.read_csv('data/src/sample.csv', header=None, encoding='shift_jis')
# 基本
df = pd.read_csv('train.csv', encoding='utf-8')

# headerがないとき (列名は連番になる)
df = pd.read_csv('train.csv', header=None)

# headerがなくて自分で列名指定したいとき
df = pd.read_csv('train.csv', names=('col_1', 'col_2'))

# 利用する列を指定したいとき
df = pd.read_csv('train.tsv', usecols=['col_1', 'col_3'], sep='\t')

# lamda式も利用可能
df = pd.read_csv('train.csv', usecols=lambda x: x is not 'col_2')

# 列名: 読み込んだあとの変更
df = df.rename(columns={'c': 'col_1'})

# 型指定で読み込み (指定した列以外は自動推定)
## メモリ逼迫しているとき以外は、型指定せず read_csv して、
## 後述の `reduce_mem_usage` を使うことも多い
df = pd.read_csv('train.csv', dtype={'col_1': str, 'col_3': str})

## 型: 読み込んだあとの変更
df = df['col_1'].astype(int)  # float / str / np.int8 ...

# 時間系データをparse
df = pd.read_csv('train.csv', parse_dates=['created_at', 'updated_at'])
```

### 書式

	df: データフレーム

| filepath_or_buffer | 読み込み元のファイルのパスや URL を指定。                                                                 |
|--------------------|-----------------------------------------------------------------------------------------------------------|
| sep                | 区切り文字。 (デフォルト: ‘,’ (read.csv) / ‘\t’ (read.table) )                                            |
| delimiter          | sep の代わりに delimiter 引数でも区切り文字を指定可能。 (デフォルト: None)                                |
| header             | ヘッダ行の行数を整数で指定。 (デフォルト: ‘infer’)                                                        |
| names              | ヘッダ行をリストで指定。 (デフォルト: None)                                                               |
| index_col          | 行のインデックスに用いる列番号。 (デフォルト: None)                                                       |
| dtype              | 各行のデータタイプ。例: {‘a’: np.float64, ‘b’: np.int32}  (デフォルト: None)                              |
| skiprows           | 先頭から読み込みをスキップする行数。 (デフォルト: None)                                                   |
| skipfooter         | 末尾から読み込みをスキップする行数。 (デフォルト: None)                                                   |
| nrows              | 読み込む行数。 (デフォルト: None)                                                                         |
| quotechar          | ダブルクォーテーションなどでクオートされている場合のクオート文字。 (デフォルト: ‘”‘)                      |
| escapechar         | エスケープされている場合のエスケープ文字。 (デフォルト: None)                                             |
| comment            | コメント行の行頭文字を指定。指定した文字で始まる行は無視されます。 (デフォルト: None)                     |
| encoding           | 文字コード。’utf-8′, ‘shift_jis’, ‘euc_jp’ などを指定。参考: Python の文字コードの一覧 (デフォルト: None) |

### 例

```python
import pandas as pd

sample4_2 = pd.read_csv("sample4.csv", header=None, names=["id","class","grade","name"])

sample4_2
```

```python
    id class  grade       name
0    1     A      1      Satou
1    3     B      1  Hashimoto
2   15     B      3  Takahashi
3  102     A      2     Aikawa
```

### 説明

