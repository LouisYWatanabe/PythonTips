# Logloss

## 書式
```python
from sklearn.metrics import log_loss

# 0, 1で表される二値分類の真の値と予測確率
y_true = [1, 0, 1, 1, 0, 1]
y_prob = [0.1, 0.2, 0.8, 0.8, 0.1, 0.3]

logloss = log_loss(y_true, y_prob)
print(logloss)
# 0.7136

# -----------------------------------
# マルチクラス分類
# -----------------------------------
# multi-class logloss

from sklearn.metrics import log_loss

# 3クラス分類の真の値と予測値
y_true = np.array([0, 2, 1, 2, 2])
y_pred = np.array([[0.68, 0.32, 0.00],
                   [0.00, 0.00, 1.00],
                   [0.60, 0.40, 0.00],
                   [0.00, 0.00, 1.00],
                   [0.28, 0.12, 0.60]])
logloss = log_loss(y_true, y_pred)
print(logloss)
# 0.3626
```

## 説明

loglossは以下の式で表される、分類タスクでの代表的な評価指標です。
cross entropyと呼ばれることもあります。

$$
log loss = -\frac{1}{N}\sum^N_{i=1}(y_ilogp_i+(1-y_i)log(1-p_i))\\
=-\frac{1}{N}\sum^N_{i=1}logp^`_i
$$
