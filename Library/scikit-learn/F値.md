# F値

```python
# テストデータで予測を行う
y_pred = tree_model.predict(X_test)

from sklearn.metrics import f1_score

# averageにデフォルトで2値分類用の'binary'が指定されているので、ここでは他の引数を設定します。
# averageにマイクロ平均（micro）を設定
print("決定木のF値 : {:.2f}".format(f1_score(y_test, y_pred, average="micro")))
```

```
決定木の再現率 : 0.93
```
