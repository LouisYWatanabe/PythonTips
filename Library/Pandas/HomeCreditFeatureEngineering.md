# 複数のテーブルの結合



## データ

教師あり分類問題で、特徴量から
- ローンを時間通りに返済する（0）
- ローンの返済が滞る（1）
の2値を分類することが目的です

- application_{train|test}.csv
    - これがメインテーブルで、Train (TARGET付き)とTest (TARGET無し)の2つのファイルに分かれています。
    - すべてのアプリケーションの静的データ。1つの行は、データサンプルの 1つのローンを表しています。
- bureau.csv
    - 顧客の他の金融機関からの過去のクレジットに関するデータ。
- bureau_balance.csv
    - クレジットビューローに報告された過去のクレジットの月次残高。
- POS_CASH_balance.csv
    - 顧客がホームクレジットで過去に受けた売却ポイントやキャッシュローンについての月次データ
- credit_card_balance.csv
    - 顧客がホームクレジットで過去に持っていたクレジットカードの月次データ。
- previous_application.csv
    - 申込データにあるローンを持つ顧客のホームクレジットでのローンの過去の申込状況。
- installments_payments.csv
    - 本サンプルのローンに関連して、ホームクレジットで過去に払い出されたクレジットの返済履歴
- HomeCredit_columns_description.csv
    - データファイルのカラムの説明
    
この図は、すべてのデータがどのように関連しているかを示しています。
![png](./image/home_credit.png)

さらに、すべてのカラムの定義(HomeCredit_columns_description.csv)と、予想される提出ファイルの例が提供されています。

基本的には`application_{train|test}.csv`のデータのみを使用します。<br>本気で高得点を望むのなら、すべてのデータを使用する必要がありますが、いったんベースラインを確立し、それに基づいてデータを改善するようにします。<br>このようなプロジェクトでは、少しずつ問題を理解していくことがベストです。

## import


```python
# データ操作
import numpy as np
import pandas as pd
# 乱数
import random
# カテゴリー変数のラベルエンコーディング
from sklearn.preprocessing import LabelEncoder
# ファイル管理
import os
import zipfile
# 警告の非表示
import warnings
warnings.filterwarnings('ignore')
# 可視化表示
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 22
%matplotlib inline
import seaborn as sns
# モデルの定義
import torch
# 自動化された特徴量エンジニアリング
import featuretools as ft
```


```python
# 乱数の設定
SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)
```

## zipファイルの解凍


```python
home_credit_dir = './data/home-credit-default-risk/'

# フォルダ''./data/titanic/'が存在しない場合にzipの解凍
if not os.path.exists(home_credit_dir):
    # home-credit-default-risk.zipを解凍
    with zipfile.ZipFile('./data/home-credit-default-risk.zip','r') as file:
        # /home-credit-default-riskディレクトリを作りその中に解凍ファイルを作製
        file.extractall(home_credit_dir)
```

## データを読み込んで小さなデータセットを作成
完全なデータセットを読み込み、SK_ID_CURRでソートし、計算を実行可能にするために最初の1000行だけを保存します。<br>後でスクリプトに変換して、データセット全体で実行することができます。

## functionの定義
reduce_mem_usageは、データのメモリを減らすためにデータ型を変更する関数です。
('reduce_mem_usage' is a functin which reduce memory usage by changing data type.) https://qiita.com/hiroyuki_kageyama/items/02865616811022f79754　を参照ください。


```python
def reduce_mem_usage(df, verbose=True):
    """
    データのメモリを減らすためにデータ型を変更する関数
    （引用元：https://www.kaggle.com/fabiendaniel/elo-world）
    （参考：https://qiita.com/hiroyuki_kageyama/items/02865616811022f79754）
    Param:
        df: DataFrame
        変換したいデータフレーム
        verbose: bool
        削減したメモリの表示
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        # columns毎に処理
        col_type = df[col].dtypes
        if col_type in numerics:
            # numericsのデータ型の範囲内のときに処理を実行. データの最大最小値を元にデータ型を効率的なものに変更
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
```


```python
# データセットを読み込み、最初の1000行に制限する(SK_ID_CURRでソート) 
# これで、実際に結果をそれなりの時間で見ることができるようになります
# この後データフレームのメモリサイズは押さえますが念のためです
PATH = './data/home-credit-default-risk'
# 利用可能なファイルリスト
print(os.listdir(PATH))

app_train = pd.read_csv(PATH+'/application_train.csv').sort_values('SK_ID_CURR').reset_index(drop = True).loc[:1000, :]
app_train = reduce_mem_usage(app_train)

app_test = pd.read_csv(PATH+'/application_test.csv').sort_values('SK_ID_CURR').reset_index(drop = True).loc[:1000, :]
app_test = reduce_mem_usage(app_test)

bureau = pd.read_csv(PATH+'/bureau.csv').sort_values(['SK_ID_CURR', 'SK_ID_BUREAU']).reset_index(drop = True).loc[:1000, :]
bureau = reduce_mem_usage(bureau)
```

    ['application_test.csv', 'application_train.csv', 'bureau.csv', 'bureau_balance.csv', 'credit_card_balance.csv', 'HomeCredit_columns_description.csv', 'installments_payments.csv', 'log_reg_baseline.csv', 'POS_CASH_balance.csv', 'previous_application.csv', 'sample_submission.csv']
    Mem. usage decreased to  0.30 Mb (67.7% reduction)
    Mem. usage decreased to  0.30 Mb (67.6% reduction)
    Mem. usage decreased to  0.06 Mb (52.2% reduction)



```python
app_train.head()
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
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>...</th>
      <th>FLAG_DOCUMENT_18</th>
      <th>FLAG_DOCUMENT_19</th>
      <th>FLAG_DOCUMENT_20</th>
      <th>FLAG_DOCUMENT_21</th>
      <th>AMT_REQ_CREDIT_BUREAU_HOUR</th>
      <th>AMT_REQ_CREDIT_BUREAU_DAY</th>
      <th>AMT_REQ_CREDIT_BUREAU_WEEK</th>
      <th>AMT_REQ_CREDIT_BUREAU_MON</th>
      <th>AMT_REQ_CREDIT_BUREAU_QRT</th>
      <th>AMT_REQ_CREDIT_BUREAU_YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>1</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 122 columns</p>
</div>




```python
bureau.head()
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
      <th>SK_ID_CURR</th>
      <th>SK_ID_BUREAU</th>
      <th>CREDIT_ACTIVE</th>
      <th>CREDIT_CURRENCY</th>
      <th>DAYS_CREDIT</th>
      <th>CREDIT_DAY_OVERDUE</th>
      <th>DAYS_CREDIT_ENDDATE</th>
      <th>DAYS_ENDDATE_FACT</th>
      <th>AMT_CREDIT_MAX_OVERDUE</th>
      <th>CNT_CREDIT_PROLONG</th>
      <th>AMT_CREDIT_SUM</th>
      <th>AMT_CREDIT_SUM_DEBT</th>
      <th>AMT_CREDIT_SUM_LIMIT</th>
      <th>AMT_CREDIT_SUM_OVERDUE</th>
      <th>CREDIT_TYPE</th>
      <th>DAYS_CREDIT_UPDATE</th>
      <th>AMT_ANNUITY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>5896630</td>
      <td>Closed</td>
      <td>currency 1</td>
      <td>-857</td>
      <td>0</td>
      <td>-492.0</td>
      <td>-553.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>112500.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-155</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>5896631</td>
      <td>Closed</td>
      <td>currency 1</td>
      <td>-909</td>
      <td>0</td>
      <td>-179.0</td>
      <td>-877.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>279720.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-155</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100001</td>
      <td>5896632</td>
      <td>Closed</td>
      <td>currency 1</td>
      <td>-879</td>
      <td>0</td>
      <td>-514.0</td>
      <td>-544.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>91620.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-155</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100001</td>
      <td>5896633</td>
      <td>Closed</td>
      <td>currency 1</td>
      <td>-1572</td>
      <td>0</td>
      <td>-1329.0</td>
      <td>-1328.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>85500.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-155</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100001</td>
      <td>5896634</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-559</td>
      <td>0</td>
      <td>902.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>337680.0</td>
      <td>113166.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-6</td>
      <td>4630.5</td>
    </tr>
  </tbody>
</table>
</div>



bureau.csvには、コンペを主催したHome Creditとは別の金融機関から提供された過去の申し込み履歴が記録されています。

訓練データとは`SK_ID_CURR`でひもづいています。<br>過去の履歴なので`app_train`の1行に対して、複数のbreau行が対応する可能性があります。

仮に1対1の関係なら、単純にデータを結合すればよく、ここでは1対Nの関係なのでN側のデータセットを何かしらの方法で集約する必要があります。<br>ここでは「過去の申し込み回数」が有効な特徴量になるという仮説を立てたとして、以下のコードで`bureau`から`SK_ID_CURR`ごとに回数を集約します


```python
# `bureau`から`SK_ID_CURR`ごとに回数を集約
previous_loss_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns={'SK_ID_BUREAU':'previous_loan_counts'})
previous_loss_counts.head()
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
      <th>SK_ID_CURR</th>
      <th>previous_loan_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100002</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100003</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100004</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



このデータを`app_train`に`SK_ID_CURR`をキーにして結合します。


```python
# データの結合
app_train = pd.merge(app_train, previous_loss_counts, on='SK_ID_CURR', how='left')
app_train.head()
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
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>...</th>
      <th>FLAG_DOCUMENT_19</th>
      <th>FLAG_DOCUMENT_20</th>
      <th>FLAG_DOCUMENT_21</th>
      <th>AMT_REQ_CREDIT_BUREAU_HOUR</th>
      <th>AMT_REQ_CREDIT_BUREAU_DAY</th>
      <th>AMT_REQ_CREDIT_BUREAU_WEEK</th>
      <th>AMT_REQ_CREDIT_BUREAU_MON</th>
      <th>AMT_REQ_CREDIT_BUREAU_QRT</th>
      <th>AMT_REQ_CREDIT_BUREAU_YEAR</th>
      <th>previous_loan_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>1</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 123 columns</p>
</div>



引数に`how='left'`を与えることで、指定したデータのうち、左側のファイルを軸にデータセットを結合するように指定します。<br>この引数を指定しないと両者に含まれる`SK_ID_CURR`データセットのみが返ります。<br>ここで`previous_loan_counts`には過去の申し込みが0回の`SK_ID_CURR`は含まれていないので、データセットに欠損が発生しています。<br>この欠損値は、意味合いから考えて0で補完するのが適切と思われます。


```python
app_train['previous_loan_counts'].fillna(0, inplace=True)
app_train.head()
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
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>...</th>
      <th>FLAG_DOCUMENT_19</th>
      <th>FLAG_DOCUMENT_20</th>
      <th>FLAG_DOCUMENT_21</th>
      <th>AMT_REQ_CREDIT_BUREAU_HOUR</th>
      <th>AMT_REQ_CREDIT_BUREAU_DAY</th>
      <th>AMT_REQ_CREDIT_BUREAU_WEEK</th>
      <th>AMT_REQ_CREDIT_BUREAU_MON</th>
      <th>AMT_REQ_CREDIT_BUREAU_QRT</th>
      <th>AMT_REQ_CREDIT_BUREAU_YEAR</th>
      <th>previous_loan_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>1</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 123 columns</p>
</div>

