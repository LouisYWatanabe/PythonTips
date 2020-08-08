# 外れ値の変換 clip

```python
import pandas as pd

train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed.csv')

# 変換する数値変数をリストに格納
num_cols = ['age', 'height', 'weight', 'amount',
            'medical_info_a1', 'medical_info_a2', 'medical_info_a3', 'medical_info_b1']

# -----------------------------------
# clipping
# -----------------------------------
# データの読み込み
train_x, test_x = load_data()
# -----------------------------------
# 列ごとに学習データの1％点、99％点を計算
p01 = train_x[num_cols].quantile(0.01)
p99 = train_x[num_cols].quantile(0.99)

# 1％点以下の値は1％点に、99％点以上の値は99％点にclippingする
train_x[num_cols] = train_x[num_cols].clip(p01, p99, axis=1)
test_x[num_cols] = test_x[num_cols].clip(p01, p99, axis=1)

```