# Label encoding and One-hot encoding

2つのユニークなカテゴリーを持つカテゴリー変数（dtype == object）にはLabel encoding。<br>つ以上のユニークなカテゴリーを持つカテゴリー変数にはOne-hot encodingを使用します。

Label encodingには`Scikit-Learn LabelEncoder`を使用し、One-hot encodingには`pd.get_dummies(df)`関数を使用します。

```python
# カテゴリー変数のラベルエンコーディング
from sklearn.preprocessing import LabelEncoder

# Label encodingのインスタンス化
le = LabelEncoder()
le_count_columns = 0    # Label encodingを実行したカラムのカウント

# 全列を確認し、Label encodingを実行
for col in app_train:
    # つのユニークなカテゴリーを持つカテゴリー変数をLabel encodingを実行
    if app_train[col].dtype == 'object':
        # もしユニークなカテゴリが2以下の場合
        if len(list(app_train[col].unique())) <= 2:
            # Label encodingの学習
            le.fit(app_train[col])
            # 訓練データとテストデータでLabel encodingの実行
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            # ラベルがエンコードされた列の数のカウント
            le_count_columns += 1
print('%d カラムをlabel encodedしました。' % le_count_columns)
```
```
3 カラムをlabel encodedしました。
```
```python
# one-hot encoding
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('訓練データの特徴量のデータ構造: ', app_train.shape)
print('テストデータの特徴量のデータ構造: ', app_test.shape)
```
```
訓練データの特徴量のデータ構造:  (307511, 243)
テストデータの特徴量のデータ構造:  (48744, 239)
```

## One-hot encodingをsklearnで実行

```
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse=False)

city_name = pd.DataFrame(shops['city_name'])
# one-hot encoding結果をnumpy配列としてを取得
city_name = enc.fit_transform(city_name)
city_name
```
```
array([[0., 0., 0., ..., 1., 0., 0.],
       [0., 0., 0., ..., 0., 0., 1.],
       [1., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 1., 0., 0.],
       [0., 0., 0., ..., 1., 0., 0.],
       [0., 0., 0., ..., 0., 1., 0.]])
```
