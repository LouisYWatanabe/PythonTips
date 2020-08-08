# 標準化（standardization）


### 書式

```python
from sklearn.preprocessing import StandardScaler

# 変換する数値変数をリストに格納
num_cols = ['age', 'height', 'weight', 'amount',
            'medical_info_a1', 'medical_info_a2', 'medical_info_a3', 'medical_info_b1']

train_x = pd.read_csv('../input/sample-data/train_preprocessed.csv').drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed.csv')

# 学習データに基づいて複数列の標準化を定義
scaler = StandardScaler()
scaler.fit(train_x[num_cols])

# 変換後のデータで各列を置換
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

```

### 説明

変数の平均を0、標準偏差を1にする操作

$$x'=\frac{x-\mu}{\sigma}$$