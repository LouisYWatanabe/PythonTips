## LightGBM

事前準備として以下の内容が必要です。

1. 学習用・検証用にデータセットを分割する
2. カテゴリー変数をリスト形式で宣言する


```python
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
data = pd.concat([train, test])
```

## カテゴリー変数を質的変数に変換


```python
# 欠損値はSとして補完
data['Embarked'].fillna(('S'), inplace=True)
# S=0 C=1 Q=2 にint型で変換
data['Embarked'] = data['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

# 性別を'male'=0, 'female'=1で変換
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}).astype(int)
```


```python
# 欠損値の確認
data.isnull().sum()
```




    PassengerId       0
    Survived        418
    Pclass            0
    Name              0
    Sex               0
    Age             263
    SibSp             0
    Parch             0
    Ticket            0
    Fare              1
    Cabin          1014
    Embarked          0
    dtype: int64



## 特徴量の追加


```python
# ParchとSibSpを足し合わせてFamilySizeを新しく作成
data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

# 家族数1の特徴量IsAloneを作成
data['IsAlone'] = 0
data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1    # 行がdata['FamilySize'] == 1のとき'IsaAlone'を1に
```


```python
# 学習に使用しないカラムリストの作成
del_colum = ['PassengerId', 'Name', 'Ticket', 'Cabin']
data.drop(del_colum, axis=1, inplace=True)
# 結合していたデータを再度訓練データとテストデータに分割
train = data[:len(train)]
test = data[len(train):]

# 目的変数と説明変数に分割
y_train = train['Survived']    # 目的変数
X_train = train.drop('Survived', axis=1)    # 訓練データの説明変数
X_test = test.drop('Survived', axis=1)    # テストデータの説明変数
```


## 学習用データを学習用・検証用に分割する
```python
# 学習用データを学習用・検証用に分割する
from sklearn.model_selection import train_test_split

# train:valid = 7:3
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,             # 対象データ1
    y_train,             # 対象データ2
    test_size=0.3,       # 検証用データを3に指定
    stratify=y_train,    # 訓練データで層化抽出
    random_state=42
)
```


```python
# カテゴリー変数をリスト形式で宣言(A-Z順で宣言する)
categorical_features = ['Embarked', 'Pclass', 'Sex']
```


## LightGBMで学習の実施
```python
# LightGBMで学習の実施
import lightgbm as lgb
# データセットの初期化
lgb_train = lgb.Dataset(
    X_train,
    y_train,
    categorical_feature=categorical_features
)

lgb_valid = lgb.Dataset(
    X_valid,
    y_valid,
    reference=lgb_train,    # 検証用データで参照として使用する訓練データの指定
    categorical_feature=categorical_features
)

# パラメータの設定
params = {
    'objective':'binary'    # logistic –バイナリ分類のロジスティック回帰 (多値分類ならmultiでsoftmax)
}

lgb_model = lgb.train(
    params,    # パラメータ
    lgb_train,    # 学習用データ
    valid_sets=[lgb_train, lgb_valid],    # 訓練中に評価されるデータ
    verbose_eval=10,    # 検証データは10個
    num_boost_round=1000,    # 学習の実行回数の最大値
    early_stopping_rounds=10    # 連続10回学習で検証データの性能が改善しない場合学習を打ち切る
)
```

    Training until validation scores don't improve for 10 rounds
    [10]	training's binary_logloss: 0.421574	valid_1's binary_logloss: 0.47201
    [20]	training's binary_logloss: 0.346677	valid_1's binary_logloss: 0.437703
    [30]	training's binary_logloss: 0.291702	valid_1's binary_logloss: 0.426322
    Early stopping, best iteration is:
    [26]	training's binary_logloss: 0.310242	valid_1's binary_logloss: 0.425192


    /opt/conda/lib/python3.7/site-packages/lightgbm/basic.py:1291: UserWarning: Using categorical_feature in Dataset.
      warnings.warn('Using categorical_feature in Dataset.')



```python
# 推論
lgb_y_pred = lgb_model.predict(
    X_test,    # 予測を行うデータ
    num_iteration=lgb_model.best_iteration, # 繰り返しのインデックス Noneの場合、best_iterationが存在するとダンプされます。それ以外の場合、すべての繰り返しがダンプされます。 <= 0の場合、すべての繰り返しがダンプされます。
)
# 結果の表示
lgb_y_pred[:10]
```




    array([0.10001485, 0.36330279, 0.10334206, 0.23411434, 0.27690778,
           0.21439238, 0.67915243, 0.19395604, 0.75603376, 0.09883529])




```python
# 予測結果の0.5を閾値として2値分類
lgb_y_pred = (lgb_y_pred > 0.5).astype(int)
# 結果の表示
lgb_y_pred[:10]
```




    array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0])




```python
# 予測データをcsvに変換
sub = pd.read_csv('../input/titanic/gender_submission.csv')    # サンプルの予測データ
sub['Survived'] = lgb_y_pred

sub.to_csv('submit_lightgbm.csv', index=False)
sub.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## ハイパーパラメーターの調整

ハイパーパラメーターのチューニングツールoptunaを使用してハイパーパラメーターを調整します。
指定方法は[optuna.trial](https://optuna.readthedocs.io/en/latest/reference/trial.html)で確認します。
- `max_bin`:各特長量の最大の分割数
- `num_leaves`:1つの決定木における分岐の末端の最大数
- `learning_rate`:テーブルデータでは一般的に学習率が低く丁寧に学習を行うほど高い性能を得られるため探索範囲は設定せず手動で調整を行うようにします。


```python
import optuna
from sklearn.metrics import log_loss    # 評価指標としてcross entropyを使用します（予測と正解の確率分布の誤差を確認）


# カテゴリー変数をリスト形式で宣言(A-Z順で宣言する)
categorical_features = ['Embarked', 'Pclass', 'Sex']

# 学習内容の定義
def objective(trial):
    # パラメータの設定
    params = {
        'objective':'binary',    # logistic –バイナリ分類のロジスティック回帰 (多値分類ならmultiでsoftmax)
        'max_bin':trial.suggest_int('max_bin', 255, 500),
        'learning_rate':0.05,
        'num_leaves': trial.suggest_int('num_leaves', 32, 128)
    }

    # データセットの初期化
    lgb_train = lgb.Dataset(
        X_train,
        y_train,
        categorical_feature=categorical_features
    )

    lgb_valid = lgb.Dataset(
        X_valid,
        y_valid,
        reference=lgb_train,    # 検証用データで参照として使用する訓練データの指定
        categorical_feature=categorical_features
    )

    model = lgb.train(
        params,    # パラメータ
        lgb_train,    # 学習用データ
        valid_sets=[lgb_train, lgb_valid],    # 訓練中に評価されるデータ
        verbose_eval=10,    # 検証データは10個
        num_boost_round=1000,    # 学習の実行回数の最大値
        early_stopping_rounds=10    # 連続10回学習で検証データの性能が改善しない場合学習を打ち切る
    )

    # 推論
    y_pred = model.predict(
        X_valid,    # 予測を行うデータ
        num_iteration=model.best_iteration, # 繰り返しのインデックス Noneの場合、best_iterationが存在するとダンプされます。それ以外の場合、すべての繰り返しがダンプされます。 <= 0の場合、すべての繰り返しがダンプされます。
    )
    # 評価
    score = log_loss(
        y_valid,    # 正解値
        y_pred      # 予測結果
    )
    return score
```


```python
# ハイパーパラメーターチューニングの実行
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
study.optimize(objective, n_trials=40)
```

    /opt/conda/lib/python3.7/site-packages/lightgbm/basic.py:1291: UserWarning: Using categorical_feature in Dataset.
      warnings.warn('Using categorical_feature in Dataset.')


    Training until validation scores don't improve for 10 rounds
    [10]	training's binary_logloss: 0.501275	valid_1's binary_logloss: 0.531742
    [20]	training's binary_logloss: 0.422075	valid_1's binary_logloss: 0.473076
    [30]	training's binary_logloss: 0.376633	valid_1's binary_logloss: 0.449098
    [40]	training's binary_logloss: 0.346575	valid_1's binary_logloss: 0.4372
    [50]	training's binary_logloss: 0.316863	valid_1's binary_logloss: 0.427911
    [60]	training's binary_logloss: 0.291938	valid_1's binary_logloss: 0.425944
    Early stopping, best iteration is:
    [56]	training's binary_logloss: 0.301316	valid_1's binary_logloss: 0.425718


    [I 2020-06-27 12:40:57,553] Finished trial#0 with value: 0.42571803807390357 with parameters: {'max_bin': 357, 'num_leaves': 83}. Best is trial#0 with value: 0.42571803807390357.
    /opt/conda/lib/python3.7/site-packages/lightgbm/basic.py:1291: UserWarning: Using categorical_feature in Dataset.
      warnings.warn('Using categorical_feature in Dataset.')


    Training until validation scores don't improve for 10 rounds
    [10]	training's binary_logloss: 0.501275	valid_1's binary_logloss: 0.531742
    [20]	training's binary_logloss: 0.422075	valid_1's binary_logloss: 0.473076
    [30]	training's binary_logloss: 0.376633	valid_1's binary_logloss: 0.449098
    [40]	training's binary_logloss: 0.346575	valid_1's binary_logloss: 0.4372
    [50]	training's binary_logloss: 0.316863	valid_1's binary_logloss: 0.427911
    [60]	training's binary_logloss: 0.291938	valid_1's binary_logloss: 0.425944
    Early stopping, best iteration is:
    [56]	training's binary_logloss: 0.301316	valid_1's binary_logloss: 0.425718

    省略

    [I 2020-06-27 12:41:13,711] Finished trial#38 with value: 0.42571803807390357 with parameters: {'max_bin': 256, 'num_leaves': 37}. Best is trial#0 with value: 0.42571803807390357.
    /opt/conda/lib/python3.7/site-packages/lightgbm/basic.py:1291: UserWarning: Using categorical_feature in Dataset.
      warnings.warn('Using categorical_feature in Dataset.')


    Training until validation scores don't improve for 10 rounds
    [10]	training's binary_logloss: 0.501275	valid_1's binary_logloss: 0.531742
    [20]	training's binary_logloss: 0.422075	valid_1's binary_logloss: 0.473076
    [30]	training's binary_logloss: 0.376633	valid_1's binary_logloss: 0.449098
    [40]	training's binary_logloss: 0.346575	valid_1's binary_logloss: 0.4372
    [50]	training's binary_logloss: 0.316863	valid_1's binary_logloss: 0.427911
    [60]	training's binary_logloss: 0.291938	valid_1's binary_logloss: 0.425944
    Early stopping, best iteration is:
    [56]	training's binary_logloss: 0.301316	valid_1's binary_logloss: 0.425718


    [I 2020-06-27 12:41:14,126] Finished trial#39 with value: 0.42571803807390357 with parameters: {'max_bin': 308, 'num_leaves': 35}. Best is trial#0 with value: 0.42571803807390357.



```python
# ベストパラメーターの表示
study.best_params
```




    {'max_bin': 357, 'num_leaves': 83}



## optunaのベストパラメーターで再度学習を行う


```python
# パラメータの設定
params = {
    'objective':'binary',    # logistic –バイナリ分類のロジスティック回帰 (多値分類ならmultiでsoftmax)
    'max_bin':study.best_params['max_bin'],
    'learning_rate':0.01,
    'num_leaves': study.best_params['num_leaves']
}

# データセットの初期化
lgb_train = lgb.Dataset(
    X_train,
    y_train,
    categorical_feature=categorical_features
)

lgb_valid = lgb.Dataset(
    X_valid,
    y_valid,
    reference=lgb_train,    # 検証用データで参照として使用する訓練データの指定
    categorical_feature=categorical_features
)

lgb_model = lgb.train(
    params,    # パラメータ
    lgb_train,    # 学習用データ
    valid_sets=[lgb_train, lgb_valid],    # 訓練中に評価されるデータ
    verbose_eval=10,    # 検証データは10個
    num_boost_round=1000,    # 学習の実行回数の最大値
    early_stopping_rounds=10    # 連続10回学習で検証データの性能が改善しない場合学習を打ち切る
)
```

    /opt/conda/lib/python3.7/site-packages/lightgbm/basic.py:1291: UserWarning: Using categorical_feature in Dataset.
      warnings.warn('Using categorical_feature in Dataset.')


    Training until validation scores don't improve for 10 rounds
    [10]	training's binary_logloss: 0.620266	valid_1's binary_logloss: 0.62804
    [20]	training's binary_logloss: 0.583084	valid_1's binary_logloss: 0.597684
    [30]	training's binary_logloss: 0.552088	valid_1's binary_logloss: 0.572673
    [40]	training's binary_logloss: 0.525791	valid_1's binary_logloss: 0.55104
    [50]	training's binary_logloss: 0.503138	valid_1's binary_logloss: 0.532086
    [60]	training's binary_logloss: 0.483654	valid_1's binary_logloss: 0.515976
    [70]	training's binary_logloss: 0.46651	valid_1's binary_logloss: 0.502773
    [80]	training's binary_logloss: 0.45089	valid_1's binary_logloss: 0.492091
    [90]	training's binary_logloss: 0.436172	valid_1's binary_logloss: 0.48253
    [100]	training's binary_logloss: 0.423271	valid_1's binary_logloss: 0.474617
    [110]	training's binary_logloss: 0.412148	valid_1's binary_logloss: 0.467667
    [120]	training's binary_logloss: 0.402289	valid_1's binary_logloss: 0.462052
    [130]	training's binary_logloss: 0.393394	valid_1's binary_logloss: 0.457526
    [140]	training's binary_logloss: 0.385249	valid_1's binary_logloss: 0.453709
    [150]	training's binary_logloss: 0.377835	valid_1's binary_logloss: 0.450708
    [160]	training's binary_logloss: 0.371059	valid_1's binary_logloss: 0.44789
    [170]	training's binary_logloss: 0.364804	valid_1's binary_logloss: 0.445756
    [180]	training's binary_logloss: 0.359061	valid_1's binary_logloss: 0.443881
    [190]	training's binary_logloss: 0.353891	valid_1's binary_logloss: 0.441928
    [200]	training's binary_logloss: 0.348057	valid_1's binary_logloss: 0.439603
    [210]	training's binary_logloss: 0.341029	valid_1's binary_logloss: 0.43642
    [220]	training's binary_logloss: 0.334562	valid_1's binary_logloss: 0.434064
    [230]	training's binary_logloss: 0.328481	valid_1's binary_logloss: 0.432373
    [240]	training's binary_logloss: 0.322894	valid_1's binary_logloss: 0.430896
    [250]	training's binary_logloss: 0.317598	valid_1's binary_logloss: 0.429603
    [260]	training's binary_logloss: 0.312338	valid_1's binary_logloss: 0.428813
    [270]	training's binary_logloss: 0.307527	valid_1's binary_logloss: 0.428734
    [280]	training's binary_logloss: 0.30276	valid_1's binary_logloss: 0.428334
    [290]	training's binary_logloss: 0.298258	valid_1's binary_logloss: 0.427778
    [300]	training's binary_logloss: 0.293916	valid_1's binary_logloss: 0.427824
    Early stopping, best iteration is:
    [299]	training's binary_logloss: 0.294354	valid_1's binary_logloss: 0.42776



```python
# 推論                 
lgb_y_pred = lgb_model.predict(
    X_test,    # 予測を行うデータ
    num_iteration=lgb_model.best_iteration, # 繰り返しのインデックス Noneの場合、best_iterationが存在するとダンプされます。それ以外の場合、すべての繰り返しがダンプされます。 <= 0の場合、すべての繰り返しがダンプされます。
)
# 予測結果の0.5を閾値として2値分類
lgb_y_pred = (lgb_y_pred > 0.5).astype(int)
```


```python
# 予測データをcsvに変換
sub = pd.read_csv('../input/titanic/gender_submission.csv')    # サンプルの予測データ
sub['Survived'] = lgb_y_pred

sub.to_csv('submit_lightgbm_optuna.csv', index=False)
sub.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 交差検証


```python
# 訓練用と検証用のデータの割合をできるだけそろえるように分割するライブラリ
from sklearn.model_selection import StratifiedKFold

y_preds = []    # 検証結果の格納先
models = []    # モデルのパラメータの格納先
oof_train = np.zeros((len(X_train), ))    # 学習で使用されなかったデータ
# 5分割して交差検証
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# カテゴリー変数をリスト形式で宣言(A-Z順で宣言する)
categorical_features = ['Embarked', 'Pclass', 'Sex']

# パラメータの設定
params = {
    'objective':'binary',    # logistic –バイナリ分類のロジスティック回帰 (多値分類ならmultiでsoftmax)
    'max_bin':study.best_params['max_bin'],
    'learning_rate':0.01,
    'num_leaves': study.best_params['num_leaves']
}

for train_index, valid_index in cv.split(X_train, y_train):
    # 訓練データを訓練データとバリデーションデータに分ける
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[valid_index]    # 分割後の訓練データ
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[valid_index]    # 分割後の検証データ
    
    # データセットの初期化
    lgb_train = lgb.Dataset(
        X_tr,
        y_tr,
        categorical_feature=categorical_features
    )
    lgb_eval = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train,    # 検証用データで参照として使用する訓練データの指定
        categorical_feature=categorical_features
    )
    
    lgb_model = lgb.train(
        params,    # パラメータ
        lgb_train,    # 学習用データ
        valid_sets=[lgb_train, lgb_eval],    # 訓練中に評価されるデータ
        verbose_eval=10,    # 検証データは10個
        num_boost_round=1000,    # 学習の実行回数の最大値
        early_stopping_rounds=10    # 連続10回学習で検証データの性能が改善しない場合学習を打ち切る
    )
    # 検証の実施
    oof_train[valid_index] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    # 予測の実施
    y_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    
    y_preds.append(y_pred)
    models.append(lgb_model)
```

    /opt/conda/lib/python3.7/site-packages/lightgbm/basic.py:1291: UserWarning: Using categorical_feature in Dataset.
      warnings.warn('Using categorical_feature in Dataset.')


    Training until validation scores don't improve for 10 rounds
    [10]	training's binary_logloss: 0.620226	valid_1's binary_logloss: 0.636086
    [20]	training's binary_logloss: 0.582844	valid_1's binary_logloss: 0.611844
    [30]	training's binary_logloss: 0.550642	valid_1's binary_logloss: 0.592282
    [40]	training's binary_logloss: 0.522831	valid_1's binary_logloss: 0.575681
    [50]	training's binary_logloss: 0.498857	valid_1's binary_logloss: 0.562789
    [60]	training's binary_logloss: 0.478177	valid_1's binary_logloss: 0.552016
    [70]	training's binary_logloss: 0.459883	valid_1's binary_logloss: 0.543196
    [80]	training's binary_logloss: 0.443728	valid_1's binary_logloss: 0.535386
    [90]	training's binary_logloss: 0.429296	valid_1's binary_logloss: 0.529242
    [100]	training's binary_logloss: 0.416766	valid_1's binary_logloss: 0.524181
    [110]	training's binary_logloss: 0.40544	valid_1's binary_logloss: 0.520051
    [120]	training's binary_logloss: 0.395172	valid_1's binary_logloss: 0.516374
    [130]	training's binary_logloss: 0.386077	valid_1's binary_logloss: 0.513471
    [140]	training's binary_logloss: 0.377778	valid_1's binary_logloss: 0.511236
    [150]	training's binary_logloss: 0.369838	valid_1's binary_logloss: 0.509837
    [160]	training's binary_logloss: 0.362646	valid_1's binary_logloss: 0.50836
    [170]	training's binary_logloss: 0.356251	valid_1's binary_logloss: 0.50667
    [180]	training's binary_logloss: 0.347843	valid_1's binary_logloss: 0.50613
    [190]	training's binary_logloss: 0.338287	valid_1's binary_logloss: 0.507107
    Early stopping, best iteration is:
    [181]	training's binary_logloss: 0.346817	valid_1's binary_logloss: 0.506128
    Training until validation scores don't improve for 10 rounds
    [10]	training's binary_logloss: 0.62185	valid_1's binary_logloss: 0.633282
    [20]	training's binary_logloss: 0.584958	valid_1's binary_logloss: 0.606782
    [30]	training's binary_logloss: 0.554193	valid_1's binary_logloss: 0.585321
    [40]	training's binary_logloss: 0.528297	valid_1's binary_logloss: 0.567492
    [50]	training's binary_logloss: 0.506334	valid_1's binary_logloss: 0.552685
    [60]	training's binary_logloss: 0.487142	valid_1's binary_logloss: 0.540022
    [70]	training's binary_logloss: 0.470325	valid_1's binary_logloss: 0.528944
    [80]	training's binary_logloss: 0.455588	valid_1's binary_logloss: 0.518991
    [90]	training's binary_logloss: 0.442486	valid_1's binary_logloss: 0.510608
    [100]	training's binary_logloss: 0.430721	valid_1's binary_logloss: 0.503295
    [110]	training's binary_logloss: 0.420249	valid_1's binary_logloss: 0.496073
    [120]	training's binary_logloss: 0.410898	valid_1's binary_logloss: 0.489673
    [130]	training's binary_logloss: 0.402385	valid_1's binary_logloss: 0.484528
    [140]	training's binary_logloss: 0.394661	valid_1's binary_logloss: 0.480444
    [150]	training's binary_logloss: 0.385551	valid_1's binary_logloss: 0.476366
    [160]	training's binary_logloss: 0.376218	valid_1's binary_logloss: 0.472898
    [170]	training's binary_logloss: 0.36812	valid_1's binary_logloss: 0.469932
    [180]	training's binary_logloss: 0.361068	valid_1's binary_logloss: 0.468317
    [190]	training's binary_logloss: 0.354774	valid_1's binary_logloss: 0.467452
    [200]	training's binary_logloss: 0.348791	valid_1's binary_logloss: 0.467512
    Early stopping, best iteration is:
    [199]	training's binary_logloss: 0.349342	valid_1's binary_logloss: 0.467225
    Training until validation scores don't improve for 10 rounds
    [10]	training's binary_logloss: 0.621067	valid_1's binary_logloss: 0.628292
    [20]	training's binary_logloss: 0.584113	valid_1's binary_logloss: 0.597024
    [30]	training's binary_logloss: 0.552819	valid_1's binary_logloss: 0.571179
    [40]	training's binary_logloss: 0.52622	valid_1's binary_logloss: 0.549999
    [50]	training's binary_logloss: 0.503522	valid_1's binary_logloss: 0.532373
    [60]	training's binary_logloss: 0.483708	valid_1's binary_logloss: 0.517867
    [70]	training's binary_logloss: 0.466144	valid_1's binary_logloss: 0.505518
    [80]	training's binary_logloss: 0.450756	valid_1's binary_logloss: 0.494658
    [90]	training's binary_logloss: 0.437209	valid_1's binary_logloss: 0.485087
    [100]	training's binary_logloss: 0.425301	valid_1's binary_logloss: 0.477697
    [110]	training's binary_logloss: 0.41439	valid_1's binary_logloss: 0.472209
    [120]	training's binary_logloss: 0.404503	valid_1's binary_logloss: 0.467362
    [130]	training's binary_logloss: 0.39531	valid_1's binary_logloss: 0.463543
    [140]	training's binary_logloss: 0.387123	valid_1's binary_logloss: 0.460783
    [150]	training's binary_logloss: 0.379684	valid_1's binary_logloss: 0.458738
    [160]	training's binary_logloss: 0.372882	valid_1's binary_logloss: 0.457247
    [170]	training's binary_logloss: 0.366809	valid_1's binary_logloss: 0.455862
    [180]	training's binary_logloss: 0.360363	valid_1's binary_logloss: 0.45392
    [190]	training's binary_logloss: 0.352352	valid_1's binary_logloss: 0.451607
    [200]	training's binary_logloss: 0.344045	valid_1's binary_logloss: 0.448862
    [210]	training's binary_logloss: 0.336766	valid_1's binary_logloss: 0.447016
    [220]	training's binary_logloss: 0.330173	valid_1's binary_logloss: 0.445244
    [230]	training's binary_logloss: 0.323518	valid_1's binary_logloss: 0.44222
    [240]	training's binary_logloss: 0.317813	valid_1's binary_logloss: 0.440109
    [250]	training's binary_logloss: 0.312032	valid_1's binary_logloss: 0.439448
    [260]	training's binary_logloss: 0.306877	valid_1's binary_logloss: 0.439178
    [270]	training's binary_logloss: 0.302066	valid_1's binary_logloss: 0.438744
    [280]	training's binary_logloss: 0.297649	valid_1's binary_logloss: 0.439175
    Early stopping, best iteration is:
    [272]	training's binary_logloss: 0.301241	valid_1's binary_logloss: 0.438609
    Training until validation scores don't improve for 10 rounds
    [10]	training's binary_logloss: 0.624637	valid_1's binary_logloss: 0.617358
    [20]	training's binary_logloss: 0.589833	valid_1's binary_logloss: 0.579271
    [30]	training's binary_logloss: 0.560654	valid_1's binary_logloss: 0.547615
    [40]	training's binary_logloss: 0.535737	valid_1's binary_logloss: 0.520602
    [50]	training's binary_logloss: 0.514364	valid_1's binary_logloss: 0.496934
    [60]	training's binary_logloss: 0.494533	valid_1's binary_logloss: 0.479023
    [70]	training's binary_logloss: 0.476906	valid_1's binary_logloss: 0.463884
    [80]	training's binary_logloss: 0.461096	valid_1's binary_logloss: 0.450985
    [90]	training's binary_logloss: 0.447397	valid_1's binary_logloss: 0.439746
    [100]	training's binary_logloss: 0.435332	valid_1's binary_logloss: 0.430612
    [110]	training's binary_logloss: 0.425073	valid_1's binary_logloss: 0.422297
    [120]	training's binary_logloss: 0.415588	valid_1's binary_logloss: 0.415601
    [130]	training's binary_logloss: 0.40726	valid_1's binary_logloss: 0.409131
    [140]	training's binary_logloss: 0.399777	valid_1's binary_logloss: 0.404293
    [150]	training's binary_logloss: 0.392743	valid_1's binary_logloss: 0.400058
    [160]	training's binary_logloss: 0.386114	valid_1's binary_logloss: 0.39595
    [170]	training's binary_logloss: 0.379937	valid_1's binary_logloss: 0.392237
    [180]	training's binary_logloss: 0.373111	valid_1's binary_logloss: 0.387013
    [190]	training's binary_logloss: 0.367081	valid_1's binary_logloss: 0.382378
    [200]	training's binary_logloss: 0.359424	valid_1's binary_logloss: 0.377935
    [210]	training's binary_logloss: 0.352782	valid_1's binary_logloss: 0.374299
    [220]	training's binary_logloss: 0.346798	valid_1's binary_logloss: 0.370723
    [230]	training's binary_logloss: 0.34101	valid_1's binary_logloss: 0.366918
    [240]	training's binary_logloss: 0.335614	valid_1's binary_logloss: 0.364611
    [250]	training's binary_logloss: 0.330576	valid_1's binary_logloss: 0.363627
    [260]	training's binary_logloss: 0.325797	valid_1's binary_logloss: 0.362757
    [270]	training's binary_logloss: 0.321505	valid_1's binary_logloss: 0.362474
    [280]	training's binary_logloss: 0.317561	valid_1's binary_logloss: 0.361475
    [290]	training's binary_logloss: 0.313618	valid_1's binary_logloss: 0.360177
    [300]	training's binary_logloss: 0.309691	valid_1's binary_logloss: 0.3592
    [310]	training's binary_logloss: 0.30623	valid_1's binary_logloss: 0.358175
    [320]	training's binary_logloss: 0.302641	valid_1's binary_logloss: 0.356768
    [330]	training's binary_logloss: 0.299198	valid_1's binary_logloss: 0.356743
    [340]	training's binary_logloss: 0.295563	valid_1's binary_logloss: 0.355748
    [350]	training's binary_logloss: 0.292117	valid_1's binary_logloss: 0.354201
    [360]	training's binary_logloss: 0.288909	valid_1's binary_logloss: 0.352704
    [370]	training's binary_logloss: 0.285914	valid_1's binary_logloss: 0.351098
    [380]	training's binary_logloss: 0.282624	valid_1's binary_logloss: 0.350362
    [390]	training's binary_logloss: 0.279738	valid_1's binary_logloss: 0.350101
    Early stopping, best iteration is:
    [384]	training's binary_logloss: 0.2814	valid_1's binary_logloss: 0.350007
    Training until validation scores don't improve for 10 rounds
    [10]	training's binary_logloss: 0.621445	valid_1's binary_logloss: 0.626571
    [20]	training's binary_logloss: 0.585394	valid_1's binary_logloss: 0.593657
    [30]	training's binary_logloss: 0.555503	valid_1's binary_logloss: 0.567021
    [40]	training's binary_logloss: 0.530492	valid_1's binary_logloss: 0.545329
    [50]	training's binary_logloss: 0.509159	valid_1's binary_logloss: 0.527077
    [60]	training's binary_logloss: 0.490505	valid_1's binary_logloss: 0.510796
    [70]	training's binary_logloss: 0.47434	valid_1's binary_logloss: 0.496912
    [80]	training's binary_logloss: 0.460304	valid_1's binary_logloss: 0.484473
    [90]	training's binary_logloss: 0.448017	valid_1's binary_logloss: 0.47372
    [100]	training's binary_logloss: 0.437138	valid_1's binary_logloss: 0.464447
    [110]	training's binary_logloss: 0.427659	valid_1's binary_logloss: 0.456751
    [120]	training's binary_logloss: 0.419335	valid_1's binary_logloss: 0.450407
    [130]	training's binary_logloss: 0.411525	valid_1's binary_logloss: 0.444611
    [140]	training's binary_logloss: 0.404251	valid_1's binary_logloss: 0.439403
    [150]	training's binary_logloss: 0.397685	valid_1's binary_logloss: 0.435021
    [160]	training's binary_logloss: 0.389725	valid_1's binary_logloss: 0.428834
    [170]	training's binary_logloss: 0.38081	valid_1's binary_logloss: 0.421882
    [180]	training's binary_logloss: 0.373078	valid_1's binary_logloss: 0.416827
    [190]	training's binary_logloss: 0.365801	valid_1's binary_logloss: 0.412337
    [200]	training's binary_logloss: 0.359236	valid_1's binary_logloss: 0.408574
    [210]	training's binary_logloss: 0.353267	valid_1's binary_logloss: 0.405637
    [220]	training's binary_logloss: 0.347682	valid_1's binary_logloss: 0.403764
    [230]	training's binary_logloss: 0.342442	valid_1's binary_logloss: 0.40171
    [240]	training's binary_logloss: 0.336648	valid_1's binary_logloss: 0.40146
    [250]	training's binary_logloss: 0.331732	valid_1's binary_logloss: 0.400504
    [260]	training's binary_logloss: 0.326996	valid_1's binary_logloss: 0.399974
    [270]	training's binary_logloss: 0.322537	valid_1's binary_logloss: 0.399206
    [280]	training's binary_logloss: 0.318369	valid_1's binary_logloss: 0.398554
    [290]	training's binary_logloss: 0.314221	valid_1's binary_logloss: 0.397561
    [300]	training's binary_logloss: 0.310069	valid_1's binary_logloss: 0.398078
    Early stopping, best iteration is:
    [293]	training's binary_logloss: 0.313067	valid_1's binary_logloss: 0.397375



```python
# 検証データをCSVファイルとして保存
pd.DataFrame(oof_train).to_csv('oof_train_skfold.csv', index=False)
print(oof_train[:10])    # 検証結果の表示
scores = [
    m.best_score['valid_1']['binary_logloss'] for m in models
]
score = sum(scores) / len(scores)

print('===CV scores===')
# 交差検証ごとの結果
print(scores)
# 交差検証の結果の平均
print(score)
```

    [0.55008614 0.02620268 0.78946965 0.02472749 0.09652052 0.97378161
     0.31356732 0.10946097 0.12331389 0.09806431]
    ===CV scores===
    [0.5061277858162259, 0.4672246363307579, 0.43860930675214355, 0.3500070845396603, 0.39737474755031466]
    0.4318687121978204



```python
from sklearn.metrics import accuracy_score

y_pred_oof = (oof_train > 0.5).astype(int)
accuracy_score(y_train, y_pred_oof)
```




    0.8250401284109149




```python
y_sub = sum(y_preds) / len(y_preds)
y_sub = (y_sub > 0.5).astype(int)
y_sub[:10]
```




    array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0])




```python
# 予測データをcsvに変換
sub = pd.read_csv('../input/titanic/gender_submission.csv')    # サンプルの予測データ

sub['Survived'] = y_sub
sub.to_csv('submission_lightgbm_skfold.csv', index=False)

sub.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


