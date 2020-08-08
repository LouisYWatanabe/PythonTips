# データのカラムを取得し、その中から特定のカラムをまとめて型変換

```python
import numpy as np
import pandas as pd
# データのカラムを取得し、その中から特定のカラムをまとめて型変換
for i in range(len(df.columns)):
    if df.columns[i] == 'job' or df.columns[i] == 'marital' or df.columns[i] == 'education' or:
        df[df.columns[i]] = df[df.columns[i]].astype(np.object)
# 型の確認
df.dtypes
```

	age           int64
	job          object
	marital      object
	education    object
	default        bool
	balance       int64
	housing        bool
	loan           bool
	contact      object
	day           int64
	month        object
	duration      int64
	campaign      int64
	pdays         int64
	previous      int64
	poutcome     object
	y              bool
	dtype: object