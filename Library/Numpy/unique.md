# 重複無しの要素抽出

```python
import numpy as np
data = [1,2,3,1,2,3,4]
np.unique(data)
```

### 書式

	data: リスト
### 例

```python
import numpy as np

data = [1,2,3,1,2,3,4]

u_data = np.unique(data)
u_data
```

```python
array([1,2,3,4])
```

### 説明

重複無しのリストを取得します