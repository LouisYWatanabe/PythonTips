# DataFrameの作成


```python
import pandas as pd
# データフレームの作成
pd.DataFrame()

```

### 書式

	pd.DataFrame( 辞書型データ, index=...)

### 引数

- <b>columns=[リスト]</b>
ソート順の指定
第二引数以降に`columns=[リスト]`とし、
リスト内の順番で表示する

- <b>index=</b>
インデックスの指定
第二引数以降に`index=[リスト]`とし、
インデックス名をリスト順にする

### 例

```python
import pandas as pd

# 辞書型を使いDataFrameを作成
df = pd.DataFrame({"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]})

print(df)
print()
print(df.head())
```

```python
       fruits  year  time
0       apple  2001     1
1      orange  2002     4
2      banana  2001     5
3  strawberry  2008     6
4   kiwifruit  2006     3

       fruits  year  time
0       apple  2001     1
1      orange  2002     4
2      banana  2001     5
3  strawberry  2008     6
4   kiwifruit  2006     3

```

### 説明

辞書型のデータ（`{key1: value1, key2: value2, ...}`）を
`columns`を指定しないで渡すと`key`で昇順にソートされる。
