# フィルタリング

```python
import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.DataFrame({"fruits":data}, index=index, )

conditions = [True, True, False, True, False]
print(series[conditions])
```

### 書式

	condition = [True, False]
	# データフレームのTrueのみを抽出する
	df[condition]

### 例

```python
import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
df = pd.Series(data, index=index)

# 値が5以上10未満の要素を含むSeriesを抽出
df = df[5 <= df][df < 10]
# df = df[(5 <= df) & (df < 10)] でもよい

print(df)
```

```
orange    5
banana    8
dtype: int64
```

```python
import numpy as np
import pandas as pd

np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# DataFrameを生成し、列を追加
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

# フィルタリングを用いて、dfの"apple"列が5以上かつ
# "kiwifruit"列が5以上の値をもつ行を含むDataFrameをdfに代入
df = df[(df["apple"] >= 5) & (df["kiwifruit"] >= 5)]

print(df)
```

```
   apple  orange  banana  strawberry  kiwifruit
1      6       8       6           3         10
5      8       2       5           4          8
8      6       8       4           8          8
```

### 説明

SeriesやDataFrameは条件式に従い`bool`型のシーケンスを返す

条件を複数つけたい時は[][]のように[]を並べる。


```python
data = {
    'name': ['John', 'Zack', 'Emily'],
    'sex': ['male', 'male', 'female'],
    'age': [22, 30, 32]
}
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
      <th>name</th>
      <th>sex</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>male</td>
      <td>22</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Zack</td>
      <td>male</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emily</td>
      <td>female</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Booleanのリストをフィルターとして使うことができる
df[[True, False, True]]
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
      <th>name</th>
      <th>sex</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>male</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emily</td>
      <td>female</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 当然SeriesでもOK
filter_series = pd.Series([True, False, True])
filter_series
```




    0     True
    1    False
    2     True
    dtype: bool




```python
df[filter_series]
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
      <th>name</th>
      <th>sex</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>male</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emily</td>
      <td>female</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.read_csv('tmdb_5000_movies.csv')
# このようにすればTrueとFalseのSeriesができる
df['original_language'] == 'ja'
```




    0       False
    1       False
    2       False
    3       False
    4       False
            ...  
    4798    False
    4799    False
    4800    False
    4801    False
    4802    False
    Name: original_language, Length: 4803, dtype: bool




```python
# 日本語の映画をフィルタ('original_language'=='ja')
df[df['original_language'] == 'ja'].head()
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
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>production_countries</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>97</th>
      <td>15000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>NaN</td>
      <td>315011</td>
      <td>[{"id": 1299, "name": "monster"}, {"id": 7671,...</td>
      <td>ja</td>
      <td>シン・ゴジラ</td>
      <td>From the mind behind Evangelion comes a hit la...</td>
      <td>9.476999</td>
      <td>[{"name": "Cine Bazar", "id": 5896}, {"name": ...</td>
      <td>[{"iso_3166_1": "JP", "name": "Japan"}]</td>
      <td>2016-07-29</td>
      <td>77000000</td>
      <td>120.0</td>
      <td>[{"iso_639_1": "it", "name": "Italiano"}, {"is...</td>
      <td>Released</td>
      <td>A god incarnate. A city doomed.</td>
      <td>Shin Godzilla</td>
      <td>6.5</td>
      <td>143</td>
    </tr>
    <tr>
      <th>1471</th>
      <td>41677699</td>
      <td>[{"id": 16, "name": "Animation"}, {"id": 10751...</td>
      <td>NaN</td>
      <td>12429</td>
      <td>[{"id": 456, "name": "mother"}, {"id": 1357, "...</td>
      <td>ja</td>
      <td>崖の上のポニョ</td>
      <td>The son of a sailor, 5-year old Sosuke lives a...</td>
      <td>39.586760</td>
      <td>[{"name": "Studio Ghibli", "id": 10342}, {"nam...</td>
      <td>[{"iso_3166_1": "JP", "name": "Japan"}]</td>
      <td>2008-07-19</td>
      <td>187479518</td>
      <td>100.0</td>
      <td>[{"iso_639_1": "ja", "name": "\u65e5\u672c\u8a...</td>
      <td>Released</td>
      <td>Welcome To A World Where Anything Is Possible.</td>
      <td>Ponyo</td>
      <td>7.5</td>
      <td>926</td>
    </tr>
    <tr>
      <th>1709</th>
      <td>30000000</td>
      <td>[{"id": 16, "name": "Animation"}, {"id": 878, ...</td>
      <td>NaN</td>
      <td>192577</td>
      <td>[]</td>
      <td>ja</td>
      <td>キャプテンハーロック</td>
      <td>Space Pirate Captain Harlock and his fearless ...</td>
      <td>14.726338</td>
      <td>[{"name": "Toei Animation Company", "id": 3116}]</td>
      <td>[{"iso_3166_1": "JP", "name": "Japan"}]</td>
      <td>2013-09-07</td>
      <td>17137302</td>
      <td>115.0</td>
      <td>[{"iso_639_1": "ja", "name": "\u65e5\u672c\u8a...</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Space Pirate Captain Harlock</td>
      <td>6.5</td>
      <td>356</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>24000000</td>
      <td>[{"id": 14, "name": "Fantasy"}, {"id": 16, "na...</td>
      <td>NaN</td>
      <td>4935</td>
      <td>[{"id": 334, "name": "flying"}, {"id": 616, "n...</td>
      <td>ja</td>
      <td>ハウルの動く城</td>
      <td>When Sophie, a shy young woman, is cursed with...</td>
      <td>49.549984</td>
      <td>[{"name": "Studio Ghibli", "id": 10342}, {"nam...</td>
      <td>[{"iso_3166_1": "JP", "name": "Japan"}]</td>
      <td>2004-11-19</td>
      <td>234710455</td>
      <td>119.0</td>
      <td>[{"iso_639_1": "ja", "name": "\u65e5\u672c\u8a...</td>
      <td>Released</td>
      <td>The two lived there</td>
      <td>Howl's Moving Castle</td>
      <td>8.2</td>
      <td>1991</td>
    </tr>
    <tr>
      <th>2247</th>
      <td>26500000</td>
      <td>[{"id": 12, "name": "Adventure"}, {"id": 14, "...</td>
      <td>NaN</td>
      <td>128</td>
      <td>[{"id": 1721, "name": "fight"}, {"id": 1994, "...</td>
      <td>ja</td>
      <td>もののけ姫</td>
      <td>Ashitaka, a prince of the disappearing Ainu tr...</td>
      <td>60.732738</td>
      <td>[{"name": "Miramax Films", "id": 14}, {"name":...</td>
      <td>[{"iso_3166_1": "JP", "name": "Japan"}]</td>
      <td>1997-07-12</td>
      <td>159375308</td>
      <td>134.0</td>
      <td>[{"iso_639_1": "ja", "name": "\u65e5\u672c\u8a...</td>
      <td>Released</td>
      <td>The Fate Of The World Rests On The Courage Of ...</td>
      <td>Princess Mononoke</td>
      <td>8.2</td>
      <td>1983</td>
    </tr>
  </tbody>
</table>
</div>




```python
#フィルタして.describe()すれば，日本映画のみの統計量がわかる
df[df['original_language']  == 'ja'].describe()
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
      <th>budget</th>
      <th>id</th>
      <th>popularity</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.600000e+01</td>
      <td>16.000000</td>
      <td>16.000000</td>
      <td>1.600000e+01</td>
      <td>16.000000</td>
      <td>16.00000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.429361e+07</td>
      <td>53894.250000</td>
      <td>25.663788</td>
      <td>6.602892e+07</td>
      <td>122.500000</td>
      <td>7.05000</td>
      <td>715.750000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.194324e+07</td>
      <td>94235.067388</td>
      <td>31.637281</td>
      <td>9.410171e+07</td>
      <td>25.690465</td>
      <td>0.95359</td>
      <td>1060.489101</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>128.000000</td>
      <td>0.212443</td>
      <td>0.000000e+00</td>
      <td>93.000000</td>
      <td>5.40000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.625000e+06</td>
      <td>341.250000</td>
      <td>4.699231</td>
      <td>2.286308e+05</td>
      <td>110.000000</td>
      <td>6.37500</td>
      <td>62.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.350000e+07</td>
      <td>10817.000000</td>
      <td>9.332925</td>
      <td>1.323115e+07</td>
      <td>119.500000</td>
      <td>7.10000</td>
      <td>140.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.100000e+07</td>
      <td>42567.250000</td>
      <td>39.629257</td>
      <td>9.759383e+07</td>
      <td>127.250000</td>
      <td>7.90000</td>
      <td>890.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.167770e+07</td>
      <td>315011.000000</td>
      <td>118.968562</td>
      <td>2.749251e+08</td>
      <td>207.000000</td>
      <td>8.30000</td>
      <td>3840.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 全体の統計量
df.describe() 
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
      <th>budget</th>
      <th>id</th>
      <th>popularity</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.803000e+03</td>
      <td>4803.000000</td>
      <td>4803.000000</td>
      <td>4.803000e+03</td>
      <td>4801.000000</td>
      <td>4803.000000</td>
      <td>4803.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.904504e+07</td>
      <td>57165.484281</td>
      <td>21.492301</td>
      <td>8.226064e+07</td>
      <td>106.875859</td>
      <td>6.092172</td>
      <td>690.217989</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.072239e+07</td>
      <td>88694.614033</td>
      <td>31.816650</td>
      <td>1.628571e+08</td>
      <td>22.611935</td>
      <td>1.194612</td>
      <td>1234.585891</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.900000e+05</td>
      <td>9014.500000</td>
      <td>4.668070</td>
      <td>0.000000e+00</td>
      <td>94.000000</td>
      <td>5.600000</td>
      <td>54.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.500000e+07</td>
      <td>14629.000000</td>
      <td>12.921594</td>
      <td>1.917000e+07</td>
      <td>103.000000</td>
      <td>6.200000</td>
      <td>235.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000e+07</td>
      <td>58610.500000</td>
      <td>28.313505</td>
      <td>9.291719e+07</td>
      <td>118.000000</td>
      <td>6.800000</td>
      <td>737.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.800000e+08</td>
      <td>459488.000000</td>
      <td>875.581305</td>
      <td>2.787965e+09</td>
      <td>338.000000</td>
      <td>10.000000</td>
      <td>13752.000000</td>
    </tr>
  </tbody>
</table>
</div>



### ()&() や ()|()で複数の条件を入れる


```python
# 日本映画でvote_average(評価スコア)が8より上 
df[(df['original_language'] == 'ja') & (df['vote_average'] > 8)].head(2)
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
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>production_countries</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1987</th>
      <td>24000000</td>
      <td>[{"id": 14, "name": "Fantasy"}, {"id": 16, "na...</td>
      <td>NaN</td>
      <td>4935</td>
      <td>[{"id": 334, "name": "flying"}, {"id": 616, "n...</td>
      <td>ja</td>
      <td>ハウルの動く城</td>
      <td>When Sophie, a shy young woman, is cursed with...</td>
      <td>49.549984</td>
      <td>[{"name": "Studio Ghibli", "id": 10342}, {"nam...</td>
      <td>[{"iso_3166_1": "JP", "name": "Japan"}]</td>
      <td>2004-11-19</td>
      <td>234710455</td>
      <td>119.0</td>
      <td>[{"iso_639_1": "ja", "name": "\u65e5\u672c\u8a...</td>
      <td>Released</td>
      <td>The two lived there</td>
      <td>Howl's Moving Castle</td>
      <td>8.2</td>
      <td>1991</td>
    </tr>
    <tr>
      <th>2247</th>
      <td>26500000</td>
      <td>[{"id": 12, "name": "Adventure"}, {"id": 14, "...</td>
      <td>NaN</td>
      <td>128</td>
      <td>[{"id": 1721, "name": "fight"}, {"id": 1994, "...</td>
      <td>ja</td>
      <td>もののけ姫</td>
      <td>Ashitaka, a prince of the disappearing Ainu tr...</td>
      <td>60.732738</td>
      <td>[{"name": "Miramax Films", "id": 14}, {"name":...</td>
      <td>[{"iso_3166_1": "JP", "name": "Japan"}]</td>
      <td>1997-07-12</td>
      <td>159375308</td>
      <td>134.0</td>
      <td>[{"iso_639_1": "ja", "name": "\u65e5\u672c\u8a...</td>
      <td>Released</td>
      <td>The Fate Of The World Rests On The Courage Of ...</td>
      <td>Princess Mononoke</td>
      <td>8.2</td>
      <td>1983</td>
    </tr>
  </tbody>
</table>
</div>




```python
#　予算が0もしくは売上が0のもの
df[(df['budget'] == 0) | (df['revenue'] == 0)].head(2)
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
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>production_countries</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>83</th>
      <td>27000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>NaN</td>
      <td>79698</td>
      <td>[]</td>
      <td>en</td>
      <td>The Lovers</td>
      <td>The Lovers is an epic romance time travel adve...</td>
      <td>2.418535</td>
      <td>[{"name": "Corsan", "id": 7299}, {"name": "Bli...</td>
      <td>[{"iso_3166_1": "AU", "name": "Australia"}, {"...</td>
      <td>2015-02-13</td>
      <td>0</td>
      <td>109.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>Love is longer than life.</td>
      <td>The Lovers</td>
      <td>4.8</td>
      <td>34</td>
    </tr>
    <tr>
      <th>135</th>
      <td>150000000</td>
      <td>[{"id": 18, "name": "Drama"}, {"id": 27, "name...</td>
      <td>http://www.thewolfmanmovie.com/</td>
      <td>7978</td>
      <td>[{"id": 494, "name": "father son relationship"...</td>
      <td>en</td>
      <td>The Wolfman</td>
      <td>Lawrence Talbot, an American man on a visit to...</td>
      <td>21.214571</td>
      <td>[{"name": "Universal Pictures", "id": 33}, {"n...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2010-02-11</td>
      <td>0</td>
      <td>102.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>When the moon is full the legend comes to life</td>
      <td>The Wolfman</td>
      <td>5.5</td>
      <td>549</td>
    </tr>
  </tbody>
</table>
</div>



### ~ (スクィグル)でNOT演算


```python
#　予算も売上も0ではない
df[~((df['budget'] == 0) | (df['revenue'] == 0))].head(2)
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
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>production_countries</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>237000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.avatarmovie.com/</td>
      <td>19995</td>
      <td>[{"id": 1463, "name": "culture clash"}, {"id":...</td>
      <td>en</td>
      <td>Avatar</td>
      <td>In the 22nd century, a paraplegic Marine is di...</td>
      <td>150.437577</td>
      <td>[{"name": "Ingenious Film Partners", "id": 289...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2009-12-10</td>
      <td>2787965087</td>
      <td>162.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300000000</td>
      <td>[{"id": 12, "name": "Adventure"}, {"id": 14, "...</td>
      <td>http://disney.go.com/disneypictures/pirates/</td>
      <td>285</td>
      <td>[{"id": 270, "name": "ocean"}, {"id": 726, "na...</td>
      <td>en</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>Captain Barbossa, long believed to be dead, ha...</td>
      <td>139.082615</td>
      <td>[{"name": "Walt Disney Pictures", "id": 2}, {"...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2007-05-19</td>
      <td>961000000</td>
      <td>169.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>At the end of the world, the adventure begins.</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>6.9</td>
      <td>4500</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 値がBooleanのカラムによく使える
data = [{'id': 'a', 'bool': True},
        {'id': 'b', 'bool': False},
        {'id': 'c', 'bool': True},]
df = pd.DataFrame(data)
```


```python
# df['カラム']を指定するだけで，フィルタになる
df[df['bool']]
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
      <th>id</th>
      <th>bool</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
# NOT
df[~df['bool']]
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
      <th>id</th>
      <th>bool</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>
