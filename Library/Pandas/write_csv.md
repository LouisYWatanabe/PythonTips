# csvの書き出し

```python
import pandas as pd

# 基本
df.to_csv('file_name.csv')

# index不要のとき (kaggle submission fileは不要なので忘れがち)
submission.to_csv('submission.csv', index=False)

# csvの書き出し
pd.to_csv('out/put/path/output.csv', encoding = 'shift-jis')
```

### 書式

	df: データフレーム

| path_or_buf | 出力するファイル名。省略した場合は、コンソール上に文字列として出力されます。                                                                                      |
|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| sep         | 区切り文字 (デフォルト: , (カンマ) )                                                                                                                              |
| index       | 行名を出力するかどうか。Falseを指定した場合、行名は出力されません。(デフォルト: True)                                                                             |
| encoding    | 出力する際の文字コード。’utf-8′, ‘shift_jis’, ‘euc_jp’ などを指定。参考: Python の文字コードの一覧 (デフォルト: Python 3 の場合 “utf-8”, Python 2 の場合 “ascii”) |

### 例

```python
import pandas as pd

sample4_2 = pd.read_csv("sample4.csv", header=None, names=["id","class","grade","name"])

sample4_2.to_csv('out/put/path/output.csv', encoding = 'shift-jis')
```

```python
    id class  grade       name
0    1     A      1      Satou
1    3     B      1  Hashimoto
2   15     B      3  Takahashi
3  102     A      2     Aikawa
```

### 説明

