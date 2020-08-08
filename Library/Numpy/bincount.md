# 要素のカウント

```python
import numpy as np
a = np.array([0, 1, 1, 0, 1, 0, 0, 0])
np.bincount(a)
```

### 書式

	a: 変数
### 例

```python
import numpy as np

a = np.array([0, 1, 2, 4 , 2, 4, 4])

# データの型の確認
# 存在しない値を0個とする
print(np.bincount(a))
```

```python
[1 1 2 0 3]
```

### 説明

配列内の要素をカウントする