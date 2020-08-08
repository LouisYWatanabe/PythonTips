このコンペティションは、「データサイエンスコンペティションに勝つ方法」Coursera コースの最終課題として提供されています。

このコンペティションでは、ロシアの最大手ソフトウェア会社の1つである1C社から提供された、毎日の売上データからなる時系列データセットを使って課題に取り組みます。

## 目的
各商品ごとの来月の販売数（item_cnt_month）を予測することです。<br>
この課題を解くことで、あなたのデータサイエンスのスキルを応用し、高めることができるでしょう。

## 評価
二乗平均誤差(RMSE)で評価されます。真の目標値は[0,20]の範囲にクリップされます。

## データ

- sales_train.csv - トレーニングセットです。2013年1月から2015年10月までの毎日の履歴データです。
- test.csv - テストセットです。これらのショップや商品の2015年11月の売上を予測する必要があります。
- sample_submission.csv - 正しい形式のサンプル提出ファイルです。
- items.csv - アイテム/商品に関する補足情報です。
- item_categories.csv - アイテムのカテゴリに関する補足情報です。
- shops.csv - ショップに関する補足情報です。

- データ構造
    - ID - テストセット内の(ショップ、アイテム)タプルを表すId
    - shop_id - ショップの一意の識別子
    - item_id - 製品の一意の識別子
    - item_category_id - アイテムカテゴリの一意の識別子。
    - item_cnt_day - 販売された製品の数。このメジャーの毎月の量を予測しています。
    - item_price - アイテムの現在の価格
    - date - 日付のフォーマット dd/mm/yyyy
    - date_block_num - 連続した月の番号で、便宜上使用します。2013年1月は0、2013年2月は1、...、2015年10月は33
    - item_name - アイテムの名前
    - shop_name - ショップ名
    - item_category_name - アイテムカテゴリの名前

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
%matplotlib inline
import seaborn as sns
# モデルの定義
from sklearn.ensemble import RandomForestClassifier
import torch
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
pred_future_dir = './data/competitive-data-science-predict-future-sales/'

# フォルダ''./data/titanic/'が存在しない場合にzipの解凍
if not os.path.exists(pred_future_dir):
    # home-credit-default-risk.zipを解凍
    with zipfile.ZipFile('./data/competitive-data-science-predict-future-sales.zip','r') as file:
        # /home-credit-default-riskディレクトリを作りその中に解凍ファイルを作製
        file.extractall(pred_future_dir)
```

## データの読み込み

利用可能なすべてのデータファイルをリストアップします。ファイルは全部で9つあります。
訓練用のメインファイル（ターゲットあり）1つ、テスト用のメインファイル（ターゲットなし）1つ、提出例ファイル1つ、各ローンに関する追加情報を含む他の6つのファイルです。


```python
PATH = './data/competitive-data-science-predict-future-sales'
# 利用可能なファイルリスト
print(os.listdir(PATH))
```

    ['items.csv', 'item_categories.csv', 'sales_train.csv', 'sample_submission.csv', 'shops.csv', 'submit_lightgbm.csv', 'submit_lightgbm_drop_futer.csv', 'test.csv']



```python
# 訓練データの読み込み
sales_train = pd.read_csv(PATH+'/sales_train.csv')
```

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
# データの読み込みとデータサイズの削減
sales_train = reduce_mem_usage(sales_train)
test = pd.read_csv(PATH+'/test.csv')
test = reduce_mem_usage(test)

sample_sub = pd.read_csv(PATH+'/sample_submission.csv')

items = pd.read_csv(PATH+'/items.csv')
items = reduce_mem_usage(items)
item_categories = pd.read_csv(PATH+'/item_categories.csv')
item_categories = reduce_mem_usage(item_categories)
shops = pd.read_csv(PATH+'/shops.csv')
shops = reduce_mem_usage(shops)
```

    Mem. usage decreased to 50.40 Mb (62.5% reduction)
    Mem. usage decreased to  1.43 Mb (70.8% reduction)
    Mem. usage decreased to  0.23 Mb (54.2% reduction)
    Mem. usage decreased to  0.00 Mb (39.9% reduction)
    Mem. usage decreased to  0.00 Mb (38.6% reduction)


## データごとの確認と特徴量の作成（特徴量の作成はpandas_profiling実行後に行っています）

順序で分析しちゃうからLabelEncoderより、one-hotencodingをしたかったけどMemoryErrorになるので微調整してます。（へぼだから解決案が...訓練データとテストデータの数が合わないとか？）


```python
# 訓練データの表示確認
print(sales_train.shape)
sales_train.head()
```

    (2935849, 6)





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
      <th>date</th>
      <th>date_block_num</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_price</th>
      <th>item_cnt_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>02.01.2013</td>
      <td>0</td>
      <td>59</td>
      <td>22154</td>
      <td>999.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>03.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2552</td>
      <td>899.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>05.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2552</td>
      <td>899.000000</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>06.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2554</td>
      <td>1709.050049</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2555</td>
      <td>1099.000000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# テストデータの表示確認
print(test.shape)
test.head()
```

    (214200, 3)





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
      <th>ID</th>
      <th>shop_id</th>
      <th>item_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5</td>
      <td>5037</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>5320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>5233</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>5</td>
      <td>5232</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>5268</td>
    </tr>
  </tbody>
</table>
</div>



testに存在してtrainにないitem_idを探します。<br>もしあった場合、これらの商品に対しての目的変数（今月の売り上げ）は予測できないので、0 にする必要があります。

この値はtestのitem_idの種類からtestのitem_idの種類とtrainのitem_idの種類の積集合を引いた値になります。
※testのitem_idの種類であり、trainのitem_idの種類である集合とtestのitem_idの種類を引けばtestのitem_idの種類以外のtrainのitem_idの種類がわかります。



```python
# testのitem_idの種類の数とtrainのitem_idの種類の共通部分の要素数を取得 test['item_id']の要素数から引かれるのでsetとして作成
test_item_id_inter = len(set(test['item_id']).intersection(set(sales_train['item_id'])))
test_item_id_len = len(set(test['item_id']))

# testに存在してtrainにないitem_idを出力
print(test_item_id_len - test_item_id_inter)
# testの商品IDの数(重複は除く)
print(test_item_id_len)
# testの総数
print(len(test))
```

    363
    5100
    214200



```python
# submitfileの表示確認
print(sample_sub.shape)
sample_sub.head()
```

    (214200, 2)





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
      <th>ID</th>
      <th>item_cnt_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(items.shape)
items.head()
```

    (22170, 3)





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
      <th>item_name</th>
      <th>item_id</th>
      <th>item_category_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>! ВО ВЛАСТИ НАВАЖДЕНИЯ (ПЛАСТ.)         D</td>
      <td>0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>!ABBYY FineReader 12 Professional Edition Full...</td>
      <td>1</td>
      <td>76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>***В ЛУЧАХ СЛАВЫ   (UNV)                    D</td>
      <td>2</td>
      <td>40</td>
    </tr>
    <tr>
      <th>3</th>
      <td>***ГОЛУБАЯ ВОЛНА  (Univ)                      D</td>
      <td>3</td>
      <td>40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>***КОРОБКА (СТЕКЛО)                       D</td>
      <td>4</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>



`item_name`はid以上に情報がなさそうなので列を削除


```python
items = items.drop(['item_name'], axis=1)
```


```python
print(item_categories.shape)
item_categories.head()
```

    (84, 2)





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
      <th>item_category_name</th>
      <th>item_category_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PC - Гарнитуры/Наушники</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Аксессуары - PS2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Аксессуары - PS3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Аксессуары - PS4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Аксессуары - PSP</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
item_categorie = item_categories['item_category_name'].unique()
item_categorie
```




    array(['PC - Гарнитуры/Наушники', 'Аксессуары - PS2', 'Аксессуары - PS3',
           'Аксессуары - PS4', 'Аксессуары - PSP', 'Аксессуары - PSVita',
           'Аксессуары - XBOX 360', 'Аксессуары - XBOX ONE', 'Билеты (Цифра)',
           'Доставка товара', 'Игровые консоли - PS2',
           'Игровые консоли - PS3', 'Игровые консоли - PS4',
           'Игровые консоли - PSP', 'Игровые консоли - PSVita',
           'Игровые консоли - XBOX 360', 'Игровые консоли - XBOX ONE',
           'Игровые консоли - Прочие', 'Игры - PS2', 'Игры - PS3',
           'Игры - PS4', 'Игры - PSP', 'Игры - PSVita', 'Игры - XBOX 360',
           'Игры - XBOX ONE', 'Игры - Аксессуары для игр',
           'Игры Android - Цифра', 'Игры MAC - Цифра',
           'Игры PC - Дополнительные издания',
           'Игры PC - Коллекционные издания', 'Игры PC - Стандартные издания',
           'Игры PC - Цифра', 'Карты оплаты (Кино, Музыка, Игры)',
           'Карты оплаты - Live!', 'Карты оплаты - Live! (Цифра)',
           'Карты оплаты - PSN', 'Карты оплаты - Windows (Цифра)',
           'Кино - Blu-Ray', 'Кино - Blu-Ray 3D', 'Кино - Blu-Ray 4K',
           'Кино - DVD', 'Кино - Коллекционное',
           'Книги - Артбуки, энциклопедии', 'Книги - Аудиокниги',
           'Книги - Аудиокниги (Цифра)', 'Книги - Аудиокниги 1С',
           'Книги - Бизнес литература', 'Книги - Комиксы, манга',
           'Книги - Компьютерная литература',
           'Книги - Методические материалы 1С', 'Книги - Открытки',
           'Книги - Познавательная литература', 'Книги - Путеводители',
           'Книги - Художественная литература', 'Книги - Цифра',
           'Музыка - CD локального производства',
           'Музыка - CD фирменного производства', 'Музыка - MP3',
           'Музыка - Винил', 'Музыка - Музыкальное видео',
           'Музыка - Подарочные издания', 'Подарки - Атрибутика',
           'Подарки - Гаджеты, роботы, спорт', 'Подарки - Мягкие игрушки',
           'Подарки - Настольные игры',
           'Подарки - Настольные игры (компактные)',
           'Подарки - Открытки, наклейки', 'Подарки - Развитие',
           'Подарки - Сертификаты, услуги', 'Подарки - Сувениры',
           'Подарки - Сувениры (в навеску)',
           'Подарки - Сумки, Альбомы, Коврики д/мыши', 'Подарки - Фигурки',
           'Программы - 1С:Предприятие 8', 'Программы - MAC (Цифра)',
           'Программы - Для дома и офиса',
           'Программы - Для дома и офиса (Цифра)', 'Программы - Обучающие',
           'Программы - Обучающие (Цифра)', 'Служебные', 'Служебные - Билеты',
           'Чистые носители (шпиль)', 'Чистые носители (штучные)',
           'Элементы питания'], dtype=object)



`item_category_name`は「タイプ-サブタイプ」の構成になっています。<br>
`type`と`subtype`を新しい特徴量として追加します。


```python
# '-'でカテゴリ名を分割
item_categories['split'] = item_categories['item_category_name'].str.split('-')
# typeには-で分割した先頭の値を代入
item_categories['type'] = item_categories['split'].map(lambda x:x[0].strip())
# sub_typeには-で分割した2番目の値を代入、sub-typeには、typeのデータをsub_typeとして代入
item_categories['sub_type'] = item_categories['split'].map(lambda x:x[1].strip() if len(x) > 1 else x[0].strip())
item_categories.head()
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
      <th>item_category_name</th>
      <th>item_category_id</th>
      <th>split</th>
      <th>type</th>
      <th>sub_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PC - Гарнитуры/Наушники</td>
      <td>0</td>
      <td>[PC ,  Гарнитуры/Наушники]</td>
      <td>PC</td>
      <td>Гарнитуры/Наушники</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Аксессуары - PS2</td>
      <td>1</td>
      <td>[Аксессуары ,  PS2]</td>
      <td>Аксессуары</td>
      <td>PS2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Аксессуары - PS3</td>
      <td>2</td>
      <td>[Аксессуары ,  PS3]</td>
      <td>Аксессуары</td>
      <td>PS3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Аксессуары - PS4</td>
      <td>3</td>
      <td>[Аксессуары ,  PS4]</td>
      <td>Аксессуары</td>
      <td>PS4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Аксессуары - PSP</td>
      <td>4</td>
      <td>[Аксессуары ,  PSP]</td>
      <td>Аксессуары</td>
      <td>PSP</td>
    </tr>
  </tbody>
</table>
</div>




```python
# splitカラムの削除
item_categories.drop('split', axis=1, inplace=True)
# 'item_category_name'カラムの削除
item_categories.drop('item_category_name', axis=1, inplace=True)
```


```python
item_categories['type'].value_counts()
```




    Книги                                13
    Подарки                              12
    Игровые консоли                       8
    Игры                                  8
    Аксессуары                            7
    Программы                             6
    Музыка                                6
    Кино                                  5
    Игры PC                               4
    Карты оплаты                          4
    Служебные                             2
    Чистые носители (штучные)             1
    Доставка товара                       1
    Игры MAC                              1
    Чистые носители (шпиль)               1
    PC                                    1
    Игры Android                          1
    Элементы питания                      1
    Карты оплаты (Кино, Музыка, Игры)     1
    Билеты (Цифра)                        1
    Name: type, dtype: int64




```python
item_categories['sub_type'].value_counts()
```




    Цифра                     4
    PS4                       3
    Blu                       3
    XBOX 360                  3
    PS3                       3
                             ..
    Гаджеты, роботы, спорт    1
    Гарнитуры/Наушники        1
    1С:Предприятие 8          1
    Комиксы, манга            1
    PSN                       1
    Name: sub_type, Length: 65, dtype: int64




```python
# # typeをone-hot encodingする
# types = pd.DataFrame(item_categories['type'])
# # one-hot encoding
# types = pd.get_dummies(types)

# # shops, city_onehotを横方向に連結
# item_categories = pd.concat([item_categories, types], axis=1)
# # shopsからcity_nameカラムを削除
# item_categories.drop('type', axis=1, inplace=True)
# item_categories.head()
```


```python
# # sub_typeをone-hot encodingする
# sub_types = pd.DataFrame(item_categories['sub_type'])
# # one-hot encoding
# sub_types = pd.get_dummies(sub_types)

# # shops, city_onehotを横方向に連結
# item_categories = pd.concat([item_categories, sub_types], axis=1)
# # shopsからcity_nameカラムを削除
# item_categories.drop('sub_type', axis=1, inplace=True)
# item_categories.head()
```


```python
# LabelEncoder
from sklearn.preprocessing import LabelEncoder

item_categories['type_code'] = LabelEncoder().fit_transform(item_categories['type'])
item_categories['subtype_code'] = LabelEncoder().fit_transform(item_categories['sub_type'])
# item_categoriesからtypeとsub_typeカラムを削除
item_categories.drop('sub_type', axis=1, inplace=True)
item_categories.drop('type', axis=1, inplace=True)
item_categories.head()
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
      <th>item_category_id</th>
      <th>type_code</th>
      <th>subtype_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(shops.shape)
shops
```

    (60, 2)





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
      <th>shop_name</th>
      <th>shop_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>!Якутск Орджоникидзе, 56 фран</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>!Якутск ТЦ "Центральный" фран</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Адыгея ТЦ "Мега"</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Балашиха ТРК "Октябрь-Киномир"</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Волжский ТЦ "Волга Молл"</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Вологда ТРЦ "Мармелад"</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Воронеж (Плехановская, 13)</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Воронеж ТРЦ "Максимир"</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Воронеж ТРЦ Сити-Парк "Град"</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Выездная Торговля</td>
      <td>9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Жуковский ул. Чкалова 39м?</td>
      <td>10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Жуковский ул. Чкалова 39м²</td>
      <td>11</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Интернет-магазин ЧС</td>
      <td>12</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Казань ТЦ "Бехетле"</td>
      <td>13</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Казань ТЦ "ПаркХаус" II</td>
      <td>14</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Калуга ТРЦ "XXI век"</td>
      <td>15</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Коломна ТЦ "Рио"</td>
      <td>16</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Красноярск ТЦ "Взлетка Плаза"</td>
      <td>17</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Красноярск ТЦ "Июнь"</td>
      <td>18</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Курск ТЦ "Пушкинский"</td>
      <td>19</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Москва "Распродажа"</td>
      <td>20</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Москва МТРЦ "Афи Молл"</td>
      <td>21</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Москва Магазин С21</td>
      <td>22</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Москва ТК "Буденовский" (пав.А2)</td>
      <td>23</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Москва ТК "Буденовский" (пав.К7)</td>
      <td>24</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Москва ТРК "Атриум"</td>
      <td>25</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Москва ТЦ "Ареал" (Беляево)</td>
      <td>26</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Москва ТЦ "МЕГА Белая Дача II"</td>
      <td>27</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Москва ТЦ "МЕГА Теплый Стан" II</td>
      <td>28</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Москва ТЦ "Новый век" (Новокосино)</td>
      <td>29</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Москва ТЦ "Перловский"</td>
      <td>30</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Москва ТЦ "Семеновский"</td>
      <td>31</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Москва ТЦ "Серебряный Дом"</td>
      <td>32</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Мытищи ТРК "XL-3"</td>
      <td>33</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Н.Новгород ТРЦ "РИО"</td>
      <td>34</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Н.Новгород ТРЦ "Фантастика"</td>
      <td>35</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Новосибирск ТРЦ "Галерея Новосибирск"</td>
      <td>36</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Новосибирск ТЦ "Мега"</td>
      <td>37</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Омск ТЦ "Мега"</td>
      <td>38</td>
    </tr>
    <tr>
      <th>39</th>
      <td>РостовНаДону ТРК "Мегацентр Горизонт"</td>
      <td>39</td>
    </tr>
    <tr>
      <th>40</th>
      <td>РостовНаДону ТРК "Мегацентр Горизонт" Островной</td>
      <td>40</td>
    </tr>
    <tr>
      <th>41</th>
      <td>РостовНаДону ТЦ "Мега"</td>
      <td>41</td>
    </tr>
    <tr>
      <th>42</th>
      <td>СПб ТК "Невский Центр"</td>
      <td>42</td>
    </tr>
    <tr>
      <th>43</th>
      <td>СПб ТК "Сенная"</td>
      <td>43</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Самара ТЦ "Мелодия"</td>
      <td>44</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Самара ТЦ "ПаркХаус"</td>
      <td>45</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Сергиев Посад ТЦ "7Я"</td>
      <td>46</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Сургут ТРЦ "Сити Молл"</td>
      <td>47</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Томск ТРЦ "Изумрудный Город"</td>
      <td>48</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Тюмень ТРЦ "Кристалл"</td>
      <td>49</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Тюмень ТЦ "Гудвин"</td>
      <td>50</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Тюмень ТЦ "Зеленый Берег"</td>
      <td>51</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Уфа ТК "Центральный"</td>
      <td>52</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Уфа ТЦ "Семья" 2</td>
      <td>53</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Химки ТЦ "Мега"</td>
      <td>54</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Цифровой склад 1С-Онлайн</td>
      <td>55</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Чехов ТРЦ "Карнавал"</td>
      <td>56</td>
    </tr>
    <tr>
      <th>57</th>
      <td>Якутск Орджоникидзе, 56</td>
      <td>57</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Якутск ТЦ "Центральный"</td>
      <td>58</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Ярославль ТЦ "Альтаир"</td>
      <td>59</td>
    </tr>
  </tbody>
</table>
</div>



shop_idで同じshop_nameをタイプミス？で登録されています。<br>重複しているshop_nameに対応するsho_idを統一します。


```python
# shop_idの統一
# マージ後を考え、`sales_train`と`test`に対してshop_id = 0 を shop_id = 57に shop_id = 1 を shop_id = 58 に shop_id = 10 を shop_id = 11に変換する
sales_train.loc[sales_train['shop_id'] == 0, 'shop_id'] = 57
test.loc[test['shop_id'] == 0, 'shop_id'] = 57

sales_train.loc[sales_train['shop_id'] == 1, 'shop_id'] = 58
test.loc[test['shop_id'] == 1, 'shop_id'] = 58

sales_train.loc[sales_train['shop_id'] == 10, 'shop_id'] = 11
test.loc[test['shop_id'] == 10, 'shop_id'] = 11
```

shop_nameはロシアの各都市名 半角スペース タイプ 半角スペース 店名のような構成です（Москваはモスクワ）。<br>最初のスペースまでを抽出し、city_nameとして追加します。（たぶん都市名が抜けているデータが何個かあるようだが今のところ無視）<br>この特徴量をOne-Hot encodingします。


```python
# shop_name先頭の!を削除
shops.loc[shops['shop_name'] == '!Якутск Орджоникидзе, 56 фран', 'shop_name'] = 'Якутск Орджоникидзе, 56 фран'
shops.loc[shops['shop_name'] == '!Якутск ТЦ "Центральный" фран', 'shop_name'] = 'кутск ТЦ "Центральный" фран'

# shop_nameの先頭を抽出してcity_nameを追加
shops['city_name'] = shops['shop_name'].str.split(' ').map(lambda x : x[0])    # 先頭から一番初めの半角スペースまでの文字列を抽出
shops.head()
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
      <th>shop_name</th>
      <th>shop_id</th>
      <th>city_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Якутск Орджоникидзе, 56 фран</td>
      <td>0</td>
      <td>Якутск</td>
    </tr>
    <tr>
      <th>1</th>
      <td>кутск ТЦ "Центральный" фран</td>
      <td>1</td>
      <td>кутск</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Адыгея ТЦ "Мега"</td>
      <td>2</td>
      <td>Адыгея</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Балашиха ТРК "Октябрь-Киномир"</td>
      <td>3</td>
      <td>Балашиха</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Волжский ТЦ "Волга Молл"</td>
      <td>4</td>
      <td>Волжский</td>
    </tr>
  </tbody>
</table>
</div>




```python
# city_name = pd.DataFrame(shops['city_name'])
# # one-hot encoding
# city_onehot = pd.get_dummies(city_name)

# # shops, city_onehotを横方向に連結
# shops = pd.concat([shops, city_onehot], axis=1)
# # shopsからcity_nameカラムを削除
# shops.drop('city_name', axis=1, inplace=True)
# shops.head()
```


```python
from sklearn.preprocessing import LabelEncoder
# LabelEncoder
shops['city_code'] = LabelEncoder().fit_transform(shops['city_name'])
# shopsからcity_nameカラムを削除
shops.drop('city_name', axis=1, inplace=True)
shops.head()
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
      <th>shop_name</th>
      <th>shop_id</th>
      <th>city_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Якутск Орджоникидзе, 56 фран</td>
      <td>0</td>
      <td>29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>кутск ТЦ "Центральный" фран</td>
      <td>1</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Адыгея ТЦ "Мега"</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Балашиха ТРК "Октябрь-Киномир"</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Волжский ТЦ "Волга Молл"</td>
      <td>4</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
shops.drop('shop_name', axis=1, inplace=True)
```

訓練データには、目的変数となる`item_cnt_month`がないので作成します。


```python
# trainデータにて、'date_block_num','shop_id','item_id'でGROUP化したDataFrameGroupByオブジェクトに対して、'item_cnt_day'を集計
group = sales_train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
# 列名の更新
group.columns = ['item_cnt_month']
# DataFrameGroupBy -> DataFrame に変換
group.reset_index(inplace=True)
group.head()
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
      <th>date_block_num</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_cnt_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2</td>
      <td>27</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
      <td>33</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>317</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2</td>
      <td>438</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2</td>
      <td>471</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# groups
groups = pd.DataFrame(group['item_cnt_month'])
# shops, city_onehotを横方向に連結
sales_train = pd.concat([sales_train, groups], axis=1)
sales_train.head()
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
      <th>date</th>
      <th>date_block_num</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_price</th>
      <th>item_cnt_day</th>
      <th>item_cnt_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>02.01.2013</td>
      <td>0</td>
      <td>59</td>
      <td>22154</td>
      <td>999.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>03.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2552</td>
      <td>899.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>05.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2552</td>
      <td>899.000000</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>06.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2554</td>
      <td>1709.050049</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2555</td>
      <td>1099.000000</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



## 欠損値の確認


```python
# 欠損値計算関数
def missing_value_table(df):
    """欠損値の数とカラムごとの割合の取得
    Param : DataFrame
    確認を行うデータフレーム
    """
    # 欠損値の合計
    mis_val = df.isnull().sum()
    # カラムごとの欠損値の割合
    mis_val_percent = 100 * mis_val / len(df)
    # 欠損値の合計と割合をテーブルに結合
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    # カラム名の編集
    mis_val_table = mis_val_table.rename(
        columns={0:'Missing Values', 1:'% of Total Values'}
    )
    # データを欠損値のあるものだけにし。小数点以下1桁表示で降順ソートする
    mis_val_table = mis_val_table[mis_val_table.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False
    ).round(1)
    
    # 欠損値をもつカラム数の表示
    print('このデータフレームのカラム数は、', df.shape[1])
    print('このデータフレームの欠損値列数は、', mis_val_table.shape[0])
    
    # 欠損値データフレームを返す
    return mis_val_table
```


```python
# 欠損値情報の表示
Missing_value = missing_value_table(sales_train)
Missing_value.head()
```

    このデータフレームのカラム数は、 7
    このデータフレームの欠損値列数は、 1





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
      <th>Missing Values</th>
      <th>% of Total Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>item_cnt_month</th>
      <td>1326725</td>
      <td>45.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
sales_train.isnull().sum()
```




    date                    0
    date_block_num          0
    shop_id                 0
    item_id                 0
    item_price              0
    item_cnt_day            0
    item_cnt_month    1326725
    dtype: int64




```python
# 欠損値情報の表示
Missing_value = missing_value_table(test)
Missing_value.head()
```

    このデータフレームのカラム数は、 3
    このデータフレームの欠損値列数は、 0





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
      <th>Missing Values</th>
      <th>% of Total Values</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
test.isnull().sum()
```




    ID         0
    shop_id    0
    item_id    0
    dtype: int64




```python
# 欠損値情報の表示
Missing_value = missing_value_table(items)
Missing_value.head()
```

    このデータフレームのカラム数は、 2
    このデータフレームの欠損値列数は、 0





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
      <th>Missing Values</th>
      <th>% of Total Values</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# 欠損値情報の表示
Missing_value = missing_value_table(item_categories)
Missing_value.head()
```

    このデータフレームのカラム数は、 3
    このデータフレームの欠損値列数は、 0





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
      <th>Missing Values</th>
      <th>% of Total Values</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# 欠損値情報の表示
Missing_value = missing_value_table(shops)
Missing_value.head()
```

    このデータフレームのカラム数は、 2
    このデータフレームの欠損値列数は、 0





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
      <th>Missing Values</th>
      <th>% of Total Values</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



## 型の確認


```python
# sales_train列ごとの型数と出現個数の確認
print('type:\n{}\n\nvalue counts:\n{}\n'.format(sales_train.dtypes, sales_train.dtypes.value_counts()))
```

    type:
    date               object
    date_block_num       int8
    shop_id              int8
    item_id             int16
    item_price        float32
    item_cnt_day      float16
    item_cnt_month    float16
    dtype: object
    
    value counts:
    int8       2
    float16    2
    object     1
    float32    1
    int16      1
    dtype: int64
    


dateは時系列に型変換が必要？


```python
# test列ごとの型数と出現個数の確認
print('type:\n{}\n\nvalue counts:\n{}\n'.format(test.dtypes, test.dtypes.value_counts()))
```

    type:
    ID         int32
    shop_id     int8
    item_id    int16
    dtype: object
    
    value counts:
    int16    1
    int32    1
    int8     1
    dtype: int64
    



```python
# items列ごとの型数と出現個数の確認
print('type:\n{}\n\nvalue counts:\n{}\n'.format(items.dtypes, items.dtypes.value_counts()))
```

    type:
    item_id             int16
    item_category_id     int8
    dtype: object
    
    value counts:
    int16    1
    int8     1
    dtype: int64
    



```python
# item_categories列ごとの型数と出現個数の確認
print('type:\n{}\n\nvalue counts:\n{}\n'.format(item_categories.dtypes, item_categories.dtypes.value_counts()))
```

    type:
    item_category_id     int8
    type_code           int64
    subtype_code        int64
    dtype: object
    
    value counts:
    int64    2
    int8     1
    dtype: int64
    



```python
# shops列ごとの型数と出現個数の確認
print('type:\n{}\n\nvalue counts:\n{}\n'.format(shops.dtypes, shops.dtypes.value_counts()))
```

    type:
    shop_id       int8
    city_code    int64
    dtype: object
    
    value counts:
    int64    1
    int8     1
    dtype: int64
    


## データの結合

sales_trainとtestそしてitemsのデータの`item_id`に対応するデータをマージします


```python
print('before ', sales_train.shape)
# データの結合
train = pd.merge(sales_train, items, on='item_id', how='left')
print('after ', train.shape)
train.head()
```

    before  (2935849, 7)
    after  (2935849, 8)





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
      <th>date</th>
      <th>date_block_num</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_price</th>
      <th>item_cnt_day</th>
      <th>item_cnt_month</th>
      <th>item_category_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>02.01.2013</td>
      <td>0</td>
      <td>59</td>
      <td>22154</td>
      <td>999.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1</th>
      <td>03.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2552</td>
      <td>899.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>58</td>
    </tr>
    <tr>
      <th>2</th>
      <td>05.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2552</td>
      <td>899.000000</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>58</td>
    </tr>
    <tr>
      <th>3</th>
      <td>06.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2554</td>
      <td>1709.050049</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>58</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2555</td>
      <td>1099.000000</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>56</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('before ', test.shape)
# データの結合
test = pd.merge(test, items, on='item_id', how='left')
print('after ', test.shape)
test.head()
```

    before  (214200, 3)
    after  (214200, 4)





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
      <th>ID</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_category_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5</td>
      <td>5037</td>
      <td>19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>5320</td>
      <td>55</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>5233</td>
      <td>19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>5</td>
      <td>5232</td>
      <td>23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>5268</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>



trainとtestをitem_categoriesの`item_category_id`と対応させるように結合


```python
print('before ', train.shape)
# データの結合
train = pd.merge(train, item_categories, on='item_category_id', how='left')
print('after ', train.shape)
train.head()
```

    before  (2935849, 8)
    after  (2935849, 10)





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
      <th>date</th>
      <th>date_block_num</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_price</th>
      <th>item_cnt_day</th>
      <th>item_cnt_month</th>
      <th>item_category_id</th>
      <th>type_code</th>
      <th>subtype_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>02.01.2013</td>
      <td>0</td>
      <td>59</td>
      <td>22154</td>
      <td>999.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>37</td>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>03.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2552</td>
      <td>899.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>58</td>
      <td>13</td>
      <td>27</td>
    </tr>
    <tr>
      <th>2</th>
      <td>05.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2552</td>
      <td>899.000000</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>58</td>
      <td>13</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>06.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2554</td>
      <td>1709.050049</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>58</td>
      <td>13</td>
      <td>27</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2555</td>
      <td>1099.000000</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>56</td>
      <td>13</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('before ', test.shape)
# データの結合
test = pd.merge(test, item_categories, on='item_category_id', how='left')
print('after ', test.shape)
test.head()
```

    before  (214200, 4)
    after  (214200, 6)





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
      <th>ID</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_category_id</th>
      <th>type_code</th>
      <th>subtype_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5</td>
      <td>5037</td>
      <td>19</td>
      <td>5</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>5320</td>
      <td>55</td>
      <td>13</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>5233</td>
      <td>19</td>
      <td>5</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>5</td>
      <td>5232</td>
      <td>23</td>
      <td>5</td>
      <td>16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>5268</td>
      <td>20</td>
      <td>5</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



trainとtestをshopsの`shop_id`と対応させるように結合


```python
print('before ', train.shape)
# データの結合
train = pd.merge(train, shops, on='shop_id', how='left')
print('after ', train.shape)
train.head()
```

    before  (2935849, 10)
    after  (2935849, 11)





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
      <th>date</th>
      <th>date_block_num</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_price</th>
      <th>item_cnt_day</th>
      <th>item_cnt_month</th>
      <th>item_category_id</th>
      <th>type_code</th>
      <th>subtype_code</th>
      <th>city_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>02.01.2013</td>
      <td>0</td>
      <td>59</td>
      <td>22154</td>
      <td>999.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>37</td>
      <td>11</td>
      <td>1</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>03.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2552</td>
      <td>899.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>58</td>
      <td>13</td>
      <td>27</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>05.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2552</td>
      <td>899.000000</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>58</td>
      <td>13</td>
      <td>27</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>06.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2554</td>
      <td>1709.050049</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>58</td>
      <td>13</td>
      <td>27</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2555</td>
      <td>1099.000000</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>56</td>
      <td>13</td>
      <td>3</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('before ', test.shape)
# データの結合
test = pd.merge(test, shops, on='shop_id', how='left')
print('after ', test.shape)
test.head()
```

    before  (214200, 6)
    after  (214200, 7)





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
      <th>ID</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_category_id</th>
      <th>type_code</th>
      <th>subtype_code</th>
      <th>city_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5</td>
      <td>5037</td>
      <td>19</td>
      <td>5</td>
      <td>10</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>5320</td>
      <td>55</td>
      <td>13</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>5233</td>
      <td>19</td>
      <td>5</td>
      <td>10</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>5</td>
      <td>5232</td>
      <td>23</td>
      <td>5</td>
      <td>16</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>5268</td>
      <td>20</td>
      <td>5</td>
      <td>11</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 欠損値情報の表示
Missing_value_train = missing_value_table(train)
Missing_value_train.head()
```

    このデータフレームのカラム数は、 11
    このデータフレームの欠損値列数は、 1





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
      <th>Missing Values</th>
      <th>% of Total Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>item_cnt_month</th>
      <td>1326725</td>
      <td>45.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 欠損値情報の表示
Missing_value_test = missing_value_table(test)
Missing_value_test.head()
```

    このデータフレームのカラム数は、 7
    このデータフレームの欠損値列数は、 0





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
      <th>Missing Values</th>
      <th>% of Total Values</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# 型の確認
train.dtypes
```




    date                 object
    date_block_num         int8
    shop_id                int8
    item_id               int16
    item_price          float32
    item_cnt_day        float16
    item_cnt_month      float16
    item_category_id       int8
    type_code             int64
    subtype_code          int64
    city_code             int64
    dtype: object




```python
# 型の確認
test.dtypes
```




    ID                  int32
    shop_id              int8
    item_id             int16
    item_category_id     int8
    type_code           int64
    subtype_code        int64
    city_code           int64
    dtype: object



item_nameからitem_priceを出してtestデータに結合する必要があるかな

## EDA

既存のデータの結合が終わったのでtrainデータのEDAを行います。<br>`pandas_profiling`は特徴量作成前にデータをマージしただけの状態で実行しています。


```python
# # 処理が重いので基本的にコメントアウト
# import pandas_profiling as pdp  # pandas_profilingのインポート
# pdp.ProfileReport(train)  # レポートの作成
```

以下の内容を後で検討したものには、○を付けています。

ただデータを結合した状態での確認内容（特徴量の追加前）
- Overview(概要)
    - shop_idは30と60のデータが多い印象です
    - item_priceには307980という外れ値がありそうです　○
    - item_cnt_dayには2168という外れ値がありそうです　○
    - item_category_idは30のデータが多い印象です
    - item_category_nameは`DVD`、`PC`、`CD`、`PS3`、`Blu-Ray`が多そうです　○
- Duplicate rowsのMost frequent
    - item_category_nameは「カテゴリ - 商品カテゴリ？」の表記方法　○
    - item_nameとitem_category_nameは`XBOX 360`の`Far Cry 3`で重複が発生しています
- Variables(特徴量の情報)
    - date_block_num 12か月目と24か月目のCountが多い　⇒ おそらくクリスマスに取引が多くなる傾向にあり、時系列がデータの予測に関係しそう
    - shop_id 31（8.0%）、25（6.3%）、54（4.9%）で全体の約19%を占めています。
    - item_priceの-1がcount 1だけどどんなデータなのか　○
    - item_category_idは40（19.2%）、30（12.0%）、55（11.6%）で全体の約40%を占めています。
    - item_category_nameは「 - 」でに分割してそれぞれ新しい特徴量としてもよいかも　○
    - shop_nameはモスクワ（Москва）見える範囲で全体の約20%を占めています。　○
    
date_block_numは全体を+1したほうが見やすい（ゼロオリジンだとデータを見た時に+1変換する必要があり、面倒）　○


```python
# date_block_numは全体を+1
train['date_block_num'] += 1
train['date_block_num']

# 2015年11月のデータのためdate_block_num = 34として列を追加
test['date_block_num'] = 34
# 型のキャスト
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)
test.head()
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
      <th>ID</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_category_id</th>
      <th>type_code</th>
      <th>subtype_code</th>
      <th>city_code</th>
      <th>date_block_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5</td>
      <td>5037</td>
      <td>19</td>
      <td>5</td>
      <td>10</td>
      <td>3</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>5320</td>
      <td>55</td>
      <td>13</td>
      <td>2</td>
      <td>3</td>
      <td>34</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>5233</td>
      <td>19</td>
      <td>5</td>
      <td>10</td>
      <td>3</td>
      <td>34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>5</td>
      <td>5232</td>
      <td>23</td>
      <td>5</td>
      <td>16</td>
      <td>3</td>
      <td>34</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>5268</td>
      <td>20</td>
      <td>5</td>
      <td>11</td>
      <td>3</td>
      <td>34</td>
    </tr>
  </tbody>
</table>
</div>



## 外れ値の確認と除外

item_priceとitem_cnt_dayの外れ値を箱ひげ図で確認します。


```python
# item_priceとitem_cnt_dayの箱ひげ図表示
plt.ylim(-100, 310000)
sns.boxplot(y='item_price', data=train)
plt.grid()
plt.tight_layout()
plt.show()
plt.ylim(-100, 2500)
plt.grid()
plt.tight_layout()
sns.boxplot(y='item_cnt_day', data=train)
```


![png](output_77_0.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x7fe56d3b1290>




![png](output_77_2.png)



```python
# item_cnt_dayを上限1200で表示
plt.ylim(-100, 1200)
plt.grid()
plt.tight_layout()
sns.boxplot(y='item_cnt_day', data=train)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fe56d350710>




![png](output_78_1.png)


#### item_priceとitem_cnt_dayの外れ値を除外します。

- item_price < 100000
- item_cnt_day < 1200


```python
# 外れ値の除外
print('before ', train.shape)
train = train[train['item_price'] < 100000]
train = train[train['item_cnt_day'] < 1200]
print('after ', train.shape)
```

    before  (2935849, 11)
    after  (2935847, 11)


#### item_priceの-1の値のデータを確認します。


```python
# item_priceが-1のデータ数
print(train[train['item_price'] == -1]['item_price'].value_counts())
# item_priceの-1の値を持つデータの確認
train[train['item_price'] == -1]
```

    -1.0    1
    Name: item_price, dtype: int64





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
      <th>date</th>
      <th>date_block_num</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_price</th>
      <th>item_cnt_day</th>
      <th>item_cnt_month</th>
      <th>item_category_id</th>
      <th>type_code</th>
      <th>subtype_code</th>
      <th>city_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>484683</th>
      <td>15.05.2013</td>
      <td>5</td>
      <td>32</td>
      <td>2973</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>19</td>
      <td>5</td>
      <td>10</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>



item_priceの-1の値と共通するほかデータ内容がないかを確認します


```python
# item_id 2973のときのitem_price出現個数の確認
train.loc[train['item_id'] == 2973, 'item_price'].value_counts()
```




     2499.000000    444
     1249.500000    124
     1249.000000     96
     1901.000000     28
     1250.000000     12
     2498.500000     12
     1562.030029     10
     1275.010010      9
     2498.750000      6
     1999.000000      3
     1329.290039      3
     1275.270020      3
     1837.849976      3
     1487.609985      3
     2498.899902      2
     1453.000000      2
     2248.800049      1
     1832.369995      1
     1049.000000      1
     1998.400024      1
     2249.000000      1
     1454.119995      1
     1248.699951      1
     1523.910034      1
     1248.900024      1
     2373.949951      1
     1249.099976      1
     1388.400024      1
     2498.399902      1
     2498.699951      1
     2498.833252      1
     2498.875000      1
     2498.916748      1
     1297.579956      1
     2427.571533      1
    -1.000000         1
    Name: item_price, dtype: int64




```python
# 円グラフで可視化
plt.figure(figsize=(8, 8))
plt.pie(
    train.loc[train['item_id'] == 2973, 'item_price'].value_counts(),    # データの出現頻度
    labels=train.loc[train['item_id'] == 2973, 'item_price'].value_counts().index,    # ラベル名の指定
    counterclock=False,    # データを時計回りに入れる
    startangle=90,          # データの開始位置 90の場合は円の上から開始
    autopct='%1.1f%%',      # グラフ内に構成割合のラベルを小数点1桁まで表示
    pctdistance=0.8         # ラベルの表示位置
)
plt.tight_layout()
plt.show()
```


![png](output_85_0.png)


なんとなく2499、1249.5、1249のどれかに変換するのがいい気がするけど感覚の世界なので保留します。<br>item_idが2973でshop_idが32のときのitem_priceを確認してみます。


```python
# item_id 2973のときの訓練データを抽出
item_id_2973 = train[train['item_id'] == 2973]
# item_id_2973の中で、shop_id 32のデータを抽出し、出現個数を確認
item_id_2973.loc[item_id_2973['shop_id']==32, 'item_price'].value_counts()
```




     2499.0    10
     1249.5     1
    -1.0        1
     1249.0     1
    Name: item_price, dtype: int64




```python
# 円グラフで可視化
plt.figure(figsize=(8, 8))
plt.pie(
    item_id_2973.loc[item_id_2973['shop_id']==32, 'item_price'].value_counts(),    # データの出現頻度
    labels=item_id_2973.loc[item_id_2973['shop_id']==32, 'item_price'].value_counts().index,    # ラベル名の指定
    counterclock=False,    # データを時計回りに入れる
    startangle=90,          # データの開始位置 90の場合は円の上から開始
    autopct='%1.1f%%',      # グラフ内に構成割合のラベルを小数点1桁まで表示
    pctdistance=0.8         # ラベルの表示位置
)
plt.tight_layout()
plt.show()
```


![png](output_88_0.png)


この結果から訓練データのitem_priceが-1のデータは2499.0で置き換えます。


```python
# item_priceの-1の値を2499.0に置き換える
train[train['item_price'] == -1] = 2499
```


```python
# 置き換えが完了したかの確認
train[train['item_price'] == -1]
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
      <th>date</th>
      <th>date_block_num</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_price</th>
      <th>item_cnt_day</th>
      <th>item_cnt_month</th>
      <th>item_category_id</th>
      <th>type_code</th>
      <th>subtype_code</th>
      <th>city_code</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
train.dtypes
```




    date                 object
    date_block_num         int8
    shop_id                int8
    item_id               int16
    item_price          float32
    item_cnt_day        float16
    item_cnt_month      float16
    item_category_id       int8
    type_code             int64
    subtype_code          int64
    city_code             int64
    dtype: object




```python
# object型の確認
train.select_dtypes(include=object).head()
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
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>02.01.2013</td>
    </tr>
    <tr>
      <th>1</th>
      <td>03.01.2013</td>
    </tr>
    <tr>
      <th>2</th>
      <td>05.01.2013</td>
    </tr>
    <tr>
      <th>3</th>
      <td>06.01.2013</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15.01.2013</td>
    </tr>
  </tbody>
</table>
</div>




```python
# object型の確認
test.select_dtypes(include=object).head()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
    </tr>
    <tr>
      <th>1</th>
    </tr>
    <tr>
      <th>2</th>
    </tr>
    <tr>
      <th>3</th>
    </tr>
    <tr>
      <th>4</th>
    </tr>
  </tbody>
</table>
</div>



## モデルのベースライン

全てのデータを結合したのでLightGBMでベースラインになるモデルを作成します。


```python
# object型の`date`を削除
train.drop('date', axis=1 ,inplace=True)
```


```python
# 説明変数item_priceがtestより多い
train.shape
```




    (2935847, 10)




```python
test.shape
```




    (214200, 8)



### 訓練データのitem_priceをテストデータにマージ

テストデータには`item_price`がないので訓練データから作成します

- 'shop_id','item_id'でグループ化し、'item_price'の平均を取得
- テストデータの'shop_id', 'item_id'にマージ
<!-- - 欠損値があれば'item_price'の中央値で補完 -->


```python
train['item_price']
```




    0           999.000000
    1           899.000000
    2           899.000000
    3          1709.050049
    4          1099.000000
                  ...     
    2935844     299.000000
    2935845     299.000000
    2935846     349.000000
    2935847     299.000000
    2935848     299.000000
    Name: item_price, Length: 2935847, dtype: float32




```python
# trainデータにて、'shop_id','item_id'でGROUP化したDataFrameGroupByオブジェクトに対して、'item_price'の平均
group_item_price = train.groupby(['shop_id','item_id']).agg({'item_price': ['mean']})
# 列名の更新
group_item_price.columns = ['item_price']
# DataFrameGroupBy -> DataFrame に変換
group_item_price.reset_index(inplace=True)
group_item_price.head()
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
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-61</td>
      <td>2499</td>
      <td>2499.00000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>27</td>
      <td>1498.50000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>30</td>
      <td>274.00000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>31</td>
      <td>626.05249</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>32</td>
      <td>146.27272</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('before ', test.shape)
# 'shop_id', 'item_id'のデータを結合
test = pd.merge(test, group_item_price, on=['shop_id', 'item_id'], how='left')
print('after ', test.shape)
test.head()
```

    before  (214200, 8)
    after  (214200, 9)





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
      <th>ID</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_category_id</th>
      <th>type_code</th>
      <th>subtype_code</th>
      <th>city_code</th>
      <th>date_block_num</th>
      <th>item_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5</td>
      <td>5037</td>
      <td>19</td>
      <td>5</td>
      <td>10</td>
      <td>3</td>
      <td>34</td>
      <td>1633.692261</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>5320</td>
      <td>55</td>
      <td>13</td>
      <td>2</td>
      <td>3</td>
      <td>34</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>5233</td>
      <td>19</td>
      <td>5</td>
      <td>10</td>
      <td>3</td>
      <td>34</td>
      <td>865.666687</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>5</td>
      <td>5232</td>
      <td>23</td>
      <td>5</td>
      <td>16</td>
      <td>3</td>
      <td>34</td>
      <td>599.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>5268</td>
      <td>20</td>
      <td>5</td>
      <td>11</td>
      <td>3</td>
      <td>34</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# item_priceの欠損値を中央値で補完
test['item_price'] = test['item_price'].fillna(test['item_price'].median())
```


```python
# 説明変数item_priceがtestより多い
train.shape
```




    (2935847, 10)




```python
test.shape
```




    (214200, 9)



## LightGBM

事前準備として以下の内容が必要です。

1. 学習用・検証用にデータセットを分割する
2. カテゴリー変数をリスト形式で宣言する（今回特になし）


```python
# 目的変数と説明変数に分割
y_train = train['item_cnt_month']    # 目的変数
X_train = train.drop('item_cnt_month', axis=1)    # 訓練データの説明変数
X_test = test
```

## 学習用データを学習用・検証用に分割する


```python
from sklearn.model_selection import train_test_split

# train:valid = 7:3
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,             # 対象データ1
    y_train,             # 対象データ2
    test_size=0.3,       # 検証用データを3に指定
#     stratify=y_train,    # 訓練データで層化抽出
    random_state=42
)
```

## LightGBMで学習の実施


```python
# LightGBMで学習の実施
import lightgbm as lgb

# カテゴリー変数をリスト形式で宣言(A-Z順で宣言する)
categorical_features = ['city_code', 'type_code', 'subtype_code']

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
        'objective': 'regression',    # 回帰問題
        'metric': 'rmse',      # RMSE (平均二乗誤差平方根) の最小化を目指す
        'learning_rate': 0.1, # 学習率
        'max_depth': -1, # 木の数 (負の値で無制限)
        'num_leaves': 9, # 枝葉の数
        'drop_rate': 0.15,
        'verbose': 0
    }

lgb_model = lgb.train(
    params,    # パラメータ
    lgb_train,    # 学習用データ
    valid_sets=[lgb_train, lgb_valid],    # 訓練中に評価されるデータ
    verbose_eval=10,    # 検証データは10個
    num_boost_round=1000,    # 学習の実行回数の最大値
    early_stopping_rounds=100    # 連続25回学習で検証データの性能が改善しない場合学習を打ち切る
)
```

    Training until validation scores don't improve for 100 rounds
    [10]	training's rmse: 6.79247	valid_1's rmse: 6.00378
    [20]	training's rmse: 6.76545	valid_1's rmse: 5.99264
    [30]	training's rmse: 6.75313	valid_1's rmse: 5.99213
    [40]	training's rmse: 6.74346	valid_1's rmse: 5.99448
    [50]	training's rmse: 6.73681	valid_1's rmse: 5.99814
    [60]	training's rmse: 6.73013	valid_1's rmse: 6.0005
    [70]	training's rmse: 6.72528	valid_1's rmse: 6.00276
    [80]	training's rmse: 6.72108	valid_1's rmse: 6.00473
    [90]	training's rmse: 6.71584	valid_1's rmse: 6.00671
    [100]	training's rmse: 6.71202	valid_1's rmse: 6.00836
    [110]	training's rmse: 6.70665	valid_1's rmse: 6.01007
    [120]	training's rmse: 6.70364	valid_1's rmse: 6.01247
    Early stopping, best iteration is:
    [26]	training's rmse: 6.75722	valid_1's rmse: 5.99163



```python
# 特徴量重要度の算出 (データフレームで取得)
cols = list(X_train.columns) # 特徴量名のリスト(目的変数CRIM以外)
f_importance = np.array(lgb_model.feature_importance()) # 特徴量重要度の算出
f_importance = f_importance / np.sum(f_importance) # 正規化(必要ない場合はコメントアウト)
df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})
df_importance = df_importance.sort_values('importance', ascending=False) # 降順ソート
display(df_importance)
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
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>date_block_num</td>
      <td>0.346154</td>
    </tr>
    <tr>
      <th>4</th>
      <td>item_cnt_day</td>
      <td>0.163462</td>
    </tr>
    <tr>
      <th>8</th>
      <td>city_code</td>
      <td>0.153846</td>
    </tr>
    <tr>
      <th>3</th>
      <td>item_price</td>
      <td>0.120192</td>
    </tr>
    <tr>
      <th>7</th>
      <td>subtype_code</td>
      <td>0.110577</td>
    </tr>
    <tr>
      <th>1</th>
      <td>shop_id</td>
      <td>0.067308</td>
    </tr>
    <tr>
      <th>2</th>
      <td>item_id</td>
      <td>0.033654</td>
    </tr>
    <tr>
      <th>5</th>
      <td>item_category_id</td>
      <td>0.004808</td>
    </tr>
    <tr>
      <th>6</th>
      <td>type_code</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 特徴量重要度の可視化
n_features = len(df_importance) # 特徴量数(説明変数の個数) 
df_plot = df_importance.sort_values('importance') # df_importanceをプロット用に特徴量重要度を昇順ソート 
f_imoprtance_plot = df_plot['importance'].values # 特徴量重要度の取得 
plt.barh(range(n_features), f_imoprtance_plot, align='center') 
cols_plot = df_plot['feature'].values # 特徴量の取得 
plt.yticks(np.arange(n_features), cols_plot)  # x軸,y軸の値の設定
plt.xlabel('Feature importance') # x軸のタイトル
plt.ylabel('Feature') # y軸のタイトル
```




    Text(0, 0.5, 'Feature')




![png](output_113_1.png)



```python
# 推論                 
lgb_y_pred = lgb_model.predict(
    X_test,    # 予測を行うデータ
    num_iteration=lgb_model.best_iteration, # 繰り返しのインデックス Noneの場合、best_iterationが存在するとダンプされます。それ以外の場合、すべての繰り返しがダンプされます。 <= 0の場合、すべての繰り返しがダンプされます。
)
# 結果の表示
lgb_y_pred[:10]
```




    array([2.11275127, 2.11275127, 2.11275127, 2.11275127, 2.11275127,
           2.26968993, 2.26968993, 2.26968993, 2.22878015, 2.20818872])




```python
# 予測データをcsvに変換
sub = pd.read_csv(PATH+'/sample_submission.csv')    # サンプルの予測データ
sub['item_cnt_month'] = lgb_y_pred

sub.to_csv(PATH+'/submit_lightgbm.csv', index=False)
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
      <th>ID</th>
      <th>item_cnt_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2.112751</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.112751</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2.112751</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2.112751</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2.112751</td>
    </tr>
  </tbody>
</table>
</div>



結果=`1.23409`

## 推論に寄与しなかった特徴量を削除してLightGBMを実行


```python
# object型の`date`を削除
train.drop(['item_category_id', 'type_code'], axis=1 ,inplace=True)
# object型の`date`を削除
test.drop(['item_category_id', 'type_code'], axis=1 ,inplace=True)
```


```python
# 目的変数と説明変数に分割
y_train = train['item_cnt_month']    # 目的変数
X_train = train.drop('item_cnt_month', axis=1)    # 訓練データの説明変数
X_test = test
```


```python
# train:valid = 7:3
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,             # 対象データ1
    y_train,             # 対象データ2
    test_size=0.3,       # 検証用データを3に指定
#     stratify=y_train,    # 訓練データで層化抽出
    random_state=42
)
```

## LightGBMで学習の実施


```python
# カテゴリー変数をリスト形式で宣言(A-Z順で宣言する)
categorical_features = ['city_code', 'subtype_code']

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
        'objective': 'regression',    # 回帰問題
        'metric': 'rmse',      # RMSE (平均二乗誤差平方根) の最小化を目指す
    }

lgb_model = lgb.train(
    params,    # パラメータ
    lgb_train,    # 学習用データ
    valid_sets=[lgb_train, lgb_valid],    # 訓練中に評価されるデータ
    verbose_eval=10,    # 検証データは10個
    num_boost_round=1000,    # 学習の実行回数の最大値
    early_stopping_rounds=25    # 連続25回学習で検証データの性能が改善しない場合学習を打ち切る
)
```

    Training until validation scores don't improve for 25 rounds
    [10]	training's rmse: 6.75436	valid_1's rmse: 5.99981
    [20]	training's rmse: 6.70464	valid_1's rmse: 5.99207
    [30]	training's rmse: 6.68266	valid_1's rmse: 5.99305
    [40]	training's rmse: 6.66415	valid_1's rmse: 5.99854
    Early stopping, best iteration is:
    [23]	training's rmse: 6.69772	valid_1's rmse: 5.99176



```python
# 特徴量重要度の算出 (データフレームで取得)
cols = list(X_train.columns) # 特徴量名のリスト(目的変数CRIM以外)
f_importance = np.array(lgb_model.feature_importance()) # 特徴量重要度の算出
f_importance = f_importance / np.sum(f_importance) # 正規化(必要ない場合はコメントアウト)
df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})
df_importance = df_importance.sort_values('importance', ascending=False) # 降順ソート
display(df_importance)
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
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>date_block_num</td>
      <td>0.226087</td>
    </tr>
    <tr>
      <th>2</th>
      <td>item_id</td>
      <td>0.159420</td>
    </tr>
    <tr>
      <th>6</th>
      <td>city_code</td>
      <td>0.144928</td>
    </tr>
    <tr>
      <th>3</th>
      <td>item_price</td>
      <td>0.142029</td>
    </tr>
    <tr>
      <th>5</th>
      <td>subtype_code</td>
      <td>0.123188</td>
    </tr>
    <tr>
      <th>1</th>
      <td>shop_id</td>
      <td>0.120290</td>
    </tr>
    <tr>
      <th>4</th>
      <td>item_cnt_day</td>
      <td>0.084058</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 特徴量重要度の可視化
n_features = len(df_importance) # 特徴量数(説明変数の個数) 
df_plot = df_importance.sort_values('importance') # df_importanceをプロット用に特徴量重要度を昇順ソート 
f_imoprtance_plot = df_plot['importance'].values # 特徴量重要度の取得 
plt.barh(range(n_features), f_imoprtance_plot, align='center') 
cols_plot = df_plot['feature'].values # 特徴量の取得 
plt.yticks(np.arange(n_features), cols_plot)  # x軸,y軸の値の設定
plt.xlabel('Feature importance') # x軸のタイトル
plt.ylabel('Feature') # y軸のタイトル
```




    Text(0, 0.5, 'Feature')




![png](output_124_1.png)



```python
## 推論                 
lgb_y_pred2 = lgb_model.predict(
    X_test,    # 予測を行うデータ
    num_iteration=lgb_model.best_iteration, # 繰り返しのインデックス Noneの場合、best_iterationが存在するとダンプされます。それ以外の場合、すべての繰り返しがダンプされます。 <= 0の場合、すべての繰り返しがダンプされます。
)
# 結果の表示
lgb_y_pred2[:10]
```




    array([2.0554154 , 2.0554154 , 2.0554154 , 2.0554154 , 2.0554154 ,
           2.22360713, 2.21943354, 2.21943354, 2.17059552, 2.10711936])




```python
# 予測データをcsvに変換
sub2 = pd.read_csv(PATH+'/sample_submission.csv')    # サンプルの予測データ
sub2['item_cnt_month'] = lgb_y_pred

sub2.to_csv(PATH+'/submit_lightgbm_drop_futer.csv', index=False)
sub2.head()
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
      <th>ID</th>
      <th>item_cnt_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2.112751</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.112751</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2.112751</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2.112751</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2.112751</td>
    </tr>
  </tbody>
</table>
</div>



結果=`1.23409`

`shop_id`や`item_id`をカテゴリー変数として学習させていないので、この辺りを変えるだけでも結果は変わるかも

## LightGBMで学習の実施


```python
# カテゴリー変数をリスト形式で宣言(A-Z順で宣言する)
categorical_features = ['city_code', 'item_id','subtype_code', 'shop_id']

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
        'objective': 'regression',    # 回帰問題
        'metric': 'rmse',      # RMSE (平均二乗誤差平方根) の最小化を目指す
    }

lgb_model = lgb.train(
    params,    # パラメータ
    lgb_train,    # 学習用データ
    valid_sets=[lgb_train, lgb_valid],    # 訓練中に評価されるデータ
    verbose_eval=10,    # 検証データは10個
    num_boost_round=1000,    # 学習の実行回数の最大値
    early_stopping_rounds=25    # 連続25回学習で検証データの性能が改善しない場合学習を打ち切る
)
```

    Training until validation scores don't improve for 25 rounds
    [10]	training's rmse: 6.68423	valid_1's rmse: 6.02726
    [20]	training's rmse: 6.58681	valid_1's rmse: 6.03769
    [30]	training's rmse: 6.50472	valid_1's rmse: 6.0487
    Early stopping, best iteration is:
    [10]	training's rmse: 6.68423	valid_1's rmse: 6.02726



```python
# 特徴量重要度の算出 (データフレームで取得)
cols = list(X_train.columns) # 特徴量名のリスト(目的変数CRIM以外)
f_importance = np.array(lgb_model.feature_importance()) # 特徴量重要度の算出
f_importance = f_importance / np.sum(f_importance) # 正規化(必要ない場合はコメントアウト)
df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})
df_importance = df_importance.sort_values('importance', ascending=False) # 降順ソート
display(df_importance)
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
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>date_block_num</td>
      <td>0.396667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>item_id</td>
      <td>0.220000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>shop_id</td>
      <td>0.173333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>item_price</td>
      <td>0.173333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>item_cnt_day</td>
      <td>0.033333</td>
    </tr>
    <tr>
      <th>6</th>
      <td>city_code</td>
      <td>0.003333</td>
    </tr>
    <tr>
      <th>5</th>
      <td>subtype_code</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 特徴量重要度の可視化
n_features = len(df_importance) # 特徴量数(説明変数の個数) 
df_plot = df_importance.sort_values('importance') # df_importanceをプロット用に特徴量重要度を昇順ソート 
f_imoprtance_plot = df_plot['importance'].values # 特徴量重要度の取得 
plt.barh(range(n_features), f_imoprtance_plot, align='center') 
cols_plot = df_plot['feature'].values # 特徴量の取得 
plt.yticks(np.arange(n_features), cols_plot)  # x軸,y軸の値の設定
plt.xlabel('Feature importance') # x軸のタイトル
plt.ylabel('Feature') # y軸のタイトル
```




    Text(0, 0.5, 'Feature')




![png](output_132_1.png)



```python
## 推論                 
lgb_y_pred3 = lgb_model.predict(
    X_test,    # 予測を行うデータ
    num_iteration=lgb_model.best_iteration, # 繰り返しのインデックス Noneの場合、best_iterationが存在するとダンプされます。それ以外の場合、すべての繰り返しがダンプされます。 <= 0の場合、すべての繰り返しがダンプされます。
)
# 結果の表示
lgb_y_pred3[:10]
```




    array([1.88436711, 1.88436711, 1.88436711, 1.88436711, 1.88436711,
           1.88436711, 1.88436711, 1.88436711, 1.88436711, 1.88436711])




```python
# 予測データをcsvに変換
sub3 = pd.read_csv(PATH+'/sample_submission.csv')    # サンプルの予測データ
sub3['item_cnt_month'] = lgb_y_pred

sub3.to_csv(PATH+'/submit_lightgbm_drop_cate.csv', index=False)
sub3.head()
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
      <th>ID</th>
      <th>item_cnt_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2.112751</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.112751</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2.112751</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2.112751</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2.112751</td>
    </tr>
  </tbody>
</table>
</div>



結果=`1.23409`
