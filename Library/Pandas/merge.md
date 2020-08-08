# データフレームの結合

```python
# 基本 (内部結合)
df = pd.merge(df, df_sub, on='key')

# 複数のカラムをキーとする
df = pd.merge(df, df_sub, on=['key_1', 'key_2'])

# 左結合
df = pd.merge(df, df_sub, on='key', how='left')

# 左右でカラム名が違うとき
df = pd.merge(df, df_sub, 
              left_on='key_left', right_on='key_right') \
       .drop('key_left', axis=1)  # キーが両方残るのでどちらか消す
```

### 書式
	[]:連結するオブジェクトの指定
	 axis: 連結方向, `=0`縦、`=1`横
	 sort:False

### 例

#### データの結合

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
    after  (2935849, 93)





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
      <th>type_PC</th>
      <th>type_Аксессуары</th>
      <th>...</th>
      <th>sub_type_Стандартные издания</th>
      <th>sub_type_Сувениры</th>
      <th>sub_type_Сувениры (в навеску)</th>
      <th>sub_type_Сумки, Альбомы, Коврики д/мыши</th>
      <th>sub_type_Фигурки</th>
      <th>sub_type_Художественная литература</th>
      <th>sub_type_Цифра</th>
      <th>sub_type_Чистые носители (шпиль)</th>
      <th>sub_type_Чистые носители (штучные)</th>
      <th>sub_type_Элементы питания</th>
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
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 93 columns</p>
</div>


#### 訓練データのitem_priceをテストデータにマージ

テストデータには`item_price`がないので訓練データから作成します

- 'shop_id','item_id'でグループ化し、'item_price'の平均を取得
- テストデータの'shop_id', 'item_id'にマージ
- 欠損値があれば'item_price'の中央値で補完

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

    before  (214200, 122)
    after  (214200, 123)





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
      <th>type_PC</th>
      <th>type_Аксессуары</th>
      <th>type_Билеты (Цифра)</th>
      <th>type_Доставка товара</th>
      <th>type_Игровые консоли</th>
      <th>type_Игры</th>
      <th>...</th>
      <th>city_name_Тюмень</th>
      <th>city_name_Уфа</th>
      <th>city_name_Химки</th>
      <th>city_name_Цифровой</th>
      <th>city_name_Чехов</th>
      <th>city_name_Якутск</th>
      <th>city_name_Ярославль</th>
      <th>city_name_кутск</th>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>1633.692261</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>5320</td>
      <td>55</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>5233</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>865.666687</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>5</td>
      <td>5232</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>599.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>5268</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 123 columns</p>
</div>




```python
# item_priceの欠損値を中央値で補完
test['item_price'] = test['item_price'].fillna(test['item_price'].median())
```


```python
# 説明変数item_priceがtestより多い
train.shape
```




    (2935847, 124)




```python
test.shape
```




    (214200, 123)

### 説明
データフレームの連結を行う
