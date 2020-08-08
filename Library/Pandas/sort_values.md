# ソート(sort_values)

```python
import pandas as pd

temp = train.groupby('sentiment').count().sort_values(by='text',
                                                      ascending=False)
temp.style.background_gradient(cmap='Purples')
```

### 書式

1. <b>df.head(欲しい行数)</b>
DataFrame先頭から欲しい行数を取得
（引数を指定しない場合先頭5行を取得）
df.head(欲しい行数)
1. 

### 例

```python
import numpy as np
import pandas as pd
# sentimentでグループ分けし、
# textで降順ソートする
temp = train.groupby('sentiment').count().sort_values(by='text',
                                                      ascending=False)
# グループ化したデータを色付け
temp.style.background_gradient(cmap='Purples')
```

```python
	textID	text	selected_text
sentiment			
neutral	11117	11117	11117
positive	8582	8582	8582
negative	7781	7781	7781
```

### 説明
