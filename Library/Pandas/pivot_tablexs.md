# ピボットテーブルの作り方


## .pivot_table()


```python
# ユーザの支払いトランザクションのテーブル
data = {'Date':['Jan-1', 'Jan-1', 'Jan-1', 'Jan-2', 'Jan-2', 'Jan-2'], 
        'User':['Emily', 'John', 'Nick', 'Kevin', 'Emily', 'John'],
        'Method':['Card', 'Card', 'Cash', 'Card', 'Cash', 'Cash'],
        'Price':[100, 250, 200, 460, 200, 130]}
df = pd.DataFrame(data)
df
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
      <th>Date</th>
      <th>User</th>
      <th>Method</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jan-1</td>
      <td>Emily</td>
      <td>Card</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jan-1</td>
      <td>John</td>
      <td>Card</td>
      <td>250</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jan-1</td>
      <td>Nick</td>
      <td>Cash</td>
      <td>200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jan-2</td>
      <td>Kevin</td>
      <td>Card</td>
      <td>460</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jan-2</td>
      <td>Emily</td>
      <td>Cash</td>
      <td>200</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Jan-2</td>
      <td>John</td>
      <td>Cash</td>
      <td>130</td>
    </tr>
  </tbody>
</table>
</div>




```python
## ピボットテーブルを作成
df.pivot_table(values='Price', index=['Date', 'User'], columns=['Method'])
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
      <th>Method</th>
      <th>Card</th>
      <th>Cash</th>
    </tr>
    <tr>
      <th>Date</th>
      <th>User</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">Jan-1</th>
      <th>Emily</th>
      <td>100.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>John</th>
      <td>250.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Nick</th>
      <td>NaN</td>
      <td>200.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Jan-2</th>
      <th>Emily</th>
      <td>NaN</td>
      <td>200.0</td>
    </tr>
    <tr>
      <th>John</th>
      <td>NaN</td>
      <td>130.0</td>
    </tr>
    <tr>
      <th>Kevin</th>
      <td>460.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# columnsとindexを入れ替えてみる
pivot_df = df.pivot_table(values='Price', index=['Date', 'Method'], columns=['User'])
pivot_df
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
      <th>User</th>
      <th>Emily</th>
      <th>John</th>
      <th>Kevin</th>
      <th>Nick</th>
    </tr>
    <tr>
      <th>Date</th>
      <th>Method</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Jan-1</th>
      <th>Card</th>
      <td>100.0</td>
      <td>250.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Cash</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>200.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Jan-2</th>
      <th>Card</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>460.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Cash</th>
      <td>200.0</td>
      <td>130.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## .xs()


```python
# cross section
# cardの行だけうまく抜き出す levelでindexのキーを指定
pivot_df.xs('Card', level='Method')
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
      <th>User</th>
      <th>Emily</th>
      <th>John</th>
      <th>Kevin</th>
      <th>Nick</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jan-1</th>
      <td>100.0</td>
      <td>250.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Jan-2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>460.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# defaultはlevel='Date'
pivot_df.xs('Jan-1')
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
      <th>User</th>
      <th>Emily</th>
      <th>John</th>
      <th>Kevin</th>
      <th>Nick</th>
    </tr>
    <tr>
      <th>Method</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Card</th>
      <td>100.0</td>
      <td>250.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Cash</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>200.0</td>
    </tr>
  </tbody>
</table>
</div>
