
```python
# DataFrameのNaNチェックに使う
df = pd.read_csv('tmdb_5000_movies.csv')
```

```python
# NaNフラグ用のカラムをつくる
df['runtime_nan'] = pd.isna(df['runtime'])
df.head(2)
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
      <th>...</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>runtime_nan</th>
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
      <td>...</td>
      <td>2009-12-10</td>
      <td>2787965087</td>
      <td>162.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
      <td>False</td>
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
      <td>...</td>
      <td>2007-05-19</td>
      <td>961000000</td>
      <td>169.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>At the end of the world, the adventure begins.</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>6.9</td>
      <td>4500</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 21 columns</p>
</div>


## .groupby


```python
# original_languageごとにグループ分けして，そのグループの各カラムの平均値
df.groupby('original_language').mean().head(2)
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
      <th>runtime_nan</th>
    </tr>
    <tr>
      <th>original_language</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>af</th>
      <td>3000000.0</td>
      <td>868.0</td>
      <td>2.504169</td>
      <td>9879971.0</td>
      <td>94.0</td>
      <td>6.9</td>
      <td>94.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ar</th>
      <td>0.0</td>
      <td>83505.0</td>
      <td>4.723389</td>
      <td>0.0</td>
      <td>92.0</td>
      <td>7.3</td>
      <td>53.5</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# ためしにjaのbudgetの平均を確認
df[df['original_language'] == 'ja']['budget'].mean()
#上の表と一致しているのがわかる
```




    14293606.1875




```python
# budgetが0のものは除いてやってみる
df[df['budget'] != 0].groupby('original_language').mean().head()
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
      <th>runtime_nan</th>
    </tr>
    <tr>
      <th>original_language</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>af</th>
      <td>3.000000e+06</td>
      <td>868.000000</td>
      <td>2.504169</td>
      <td>9.879971e+06</td>
      <td>94.000000</td>
      <td>6.900000</td>
      <td>94.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>cn</th>
      <td>1.838814e+07</td>
      <td>107731.857143</td>
      <td>11.497406</td>
      <td>4.497195e+07</td>
      <td>108.142857</td>
      <td>6.642857</td>
      <td>256.428571</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>da</th>
      <td>1.136667e+07</td>
      <td>31008.333333</td>
      <td>20.070381</td>
      <td>3.488204e+07</td>
      <td>120.666667</td>
      <td>7.016667</td>
      <td>514.666667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>de</th>
      <td>1.454010e+07</td>
      <td>69160.437500</td>
      <td>14.612254</td>
      <td>2.339749e+07</td>
      <td>115.000000</td>
      <td>6.725000</td>
      <td>380.187500</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>en</th>
      <td>3.820497e+07</td>
      <td>49290.748047</td>
      <td>26.549415</td>
      <td>1.077954e+08</td>
      <td>109.029026</td>
      <td>6.200837</td>
      <td>881.667690</td>
      <td>0.000279</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 各original_languageのグループにどれだけいくつbudget=0のものがあったのか
df[df['budget'] == 0].groupby('original_language').count()['budget'][:5]
```




    original_language
    ar     2
    cn     5
    cs     2
    da     1
    de    11
    Name: budget, dtype: int64




```python
#指定したグループの各カラムのdescribe()を一気に取得 (かなり見にくい)
df.groupby('original_language').describe().head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">budget</th>
      <th colspan="2" halign="left">id</th>
      <th>...</th>
      <th colspan="2" halign="left">vote_average</th>
      <th colspan="8" halign="left">vote_count</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>...</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>original_language</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>af</th>
      <td>1.0</td>
      <td>3.000000e+06</td>
      <td>NaN</td>
      <td>3000000.0</td>
      <td>3000000.0</td>
      <td>3000000.0</td>
      <td>3000000.0</td>
      <td>3000000.0</td>
      <td>1.0</td>
      <td>868.000000</td>
      <td>...</td>
      <td>6.900</td>
      <td>6.9</td>
      <td>1.0</td>
      <td>94.000000</td>
      <td>NaN</td>
      <td>94.0</td>
      <td>94.00</td>
      <td>94.0</td>
      <td>94.00</td>
      <td>94.0</td>
    </tr>
    <tr>
      <th>ar</th>
      <td>2.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>83505.000000</td>
      <td>...</td>
      <td>7.550</td>
      <td>7.8</td>
      <td>2.0</td>
      <td>53.500000</td>
      <td>2.121320</td>
      <td>52.0</td>
      <td>52.75</td>
      <td>53.5</td>
      <td>54.25</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>cn</th>
      <td>12.0</td>
      <td>1.072641e+07</td>
      <td>1.245921e+07</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6000000.0</td>
      <td>20750000.0</td>
      <td>36000000.0</td>
      <td>12.0</td>
      <td>109470.500000</td>
      <td>...</td>
      <td>6.700</td>
      <td>7.2</td>
      <td>12.0</td>
      <td>220.916667</td>
      <td>243.485470</td>
      <td>5.0</td>
      <td>54.25</td>
      <td>121.0</td>
      <td>276.25</td>
      <td>831.0</td>
    </tr>
    <tr>
      <th>cs</th>
      <td>2.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>185196.000000</td>
      <td>...</td>
      <td>6.325</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>12.000000</td>
      <td>14.142136</td>
      <td>2.0</td>
      <td>7.00</td>
      <td>12.0</td>
      <td>17.00</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>da</th>
      <td>7.0</td>
      <td>9.742857e+06</td>
      <td>1.207834e+07</td>
      <td>0.0</td>
      <td>2550000.0</td>
      <td>6500000.0</td>
      <td>10800000.0</td>
      <td>35000000.0</td>
      <td>7.0</td>
      <td>33440.714286</td>
      <td>...</td>
      <td>7.750</td>
      <td>7.9</td>
      <td>7.0</td>
      <td>450.428571</td>
      <td>434.369987</td>
      <td>65.0</td>
      <td>73.00</td>
      <td>207.0</td>
      <td>904.00</td>
      <td>927.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 56 columns</p>
</div>




```python
# revenue部分だけみる
df.groupby('original_language').describe()['revenue'].head()
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>original_language</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>af</th>
      <td>1.0</td>
      <td>9.879971e+06</td>
      <td>NaN</td>
      <td>9879971.0</td>
      <td>9879971.0</td>
      <td>9879971.0</td>
      <td>9879971.00</td>
      <td>9879971.0</td>
    </tr>
    <tr>
      <th>ar</th>
      <td>2.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>cn</th>
      <td>12.0</td>
      <td>3.374016e+07</td>
      <td>4.920072e+07</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12326197.0</td>
      <td>43907937.75</td>
      <td>156844753.0</td>
    </tr>
    <tr>
      <th>cs</th>
      <td>2.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>da</th>
      <td>7.0</td>
      <td>2.989889e+07</td>
      <td>6.245143e+07</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>674918.0</td>
      <td>19308649.00</td>
      <td>170000000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 各グループのrevenueの最大値
df.groupby('original_language')['revenue'].max()[:5]
```




    original_language
    af      9879971
    ar            0
    cn    156844753
    cs            0
    da    170000000
    Name: revenue, dtype: int64




```python
#revenueが最大のレコードのindexを取得
df.groupby('original_language')['revenue'].idxmax()[:5]
```




    original_language
    af    3936
    ar    4164
    cn    1357
    cs    2906
    da    4586
    Name: revenue, dtype: int64




```python
# このindexのSeriesを使って，レコード表示
df.iloc[df.groupby('original_language')['revenue'].idxmax()].head()
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
      <th>...</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>runtime_nan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3936</th>
      <td>3000000</td>
      <td>[{"id": 80, "name": "Crime"}, {"id": 18, "name...</td>
      <td>http://www.tsotsimovie.com/</td>
      <td>868</td>
      <td>[{"id": 100, "name": "slum"}, {"id": 1009, "na...</td>
      <td>af</td>
      <td>Tsotsi</td>
      <td>The South African multi-award winning film abo...</td>
      <td>2.504169</td>
      <td>[{"name": "Movieworld Productions", "id": 563}...</td>
      <td>...</td>
      <td>2005-08-18</td>
      <td>9879971</td>
      <td>94.0</td>
      <td>[{"iso_639_1": "af", "name": "Afrikaans"}, {"i...</td>
      <td>Released</td>
      <td>In this world... Redemption comes just once.</td>
      <td>Tsotsi</td>
      <td>6.9</td>
      <td>94</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4164</th>
      <td>0</td>
      <td>[{"id": 99, "name": "Documentary"}, {"id": 18,...</td>
      <td>http://www.thesquarefilm.com/</td>
      <td>159037</td>
      <td>[{"id": 187056, "name": "woman director"}]</td>
      <td>ar</td>
      <td>The Square</td>
      <td>The Square, a new film by Jehane Noujaim (Cont...</td>
      <td>4.892203</td>
      <td>[{"name": "Roast Beef Productions", "id": 7266...</td>
      <td>...</td>
      <td>2013-06-07</td>
      <td>0</td>
      <td>88.0</td>
      <td>[{"iso_639_1": "ar", "name": "\u0627\u0644\u06...</td>
      <td>Released</td>
      <td>What does it mean to risk your life for your i...</td>
      <td>The Square</td>
      <td>7.8</td>
      <td>55</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1357</th>
      <td>36000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 18, "nam...</td>
      <td>NaN</td>
      <td>365222</td>
      <td>[{"id": 5565, "name": "biography"}]</td>
      <td>cn</td>
      <td>葉問3</td>
      <td>When a band of brutal gangsters led by a crook...</td>
      <td>19.167377</td>
      <td>[{"name": "Mandarin Films Distribution Co.", "...</td>
      <td>...</td>
      <td>2015-12-19</td>
      <td>156844753</td>
      <td>105.0</td>
      <td>[{"iso_639_1": "cn", "name": "\u5e7f\u5dde\u8b...</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Ip Man 3</td>
      <td>6.5</td>
      <td>379</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2906</th>
      <td>0</td>
      <td>[{"id": 18, "name": "Drama"}, {"id": 10749, "n...</td>
      <td>https://www.facebook.com/eastsidestorymovie</td>
      <td>357837</td>
      <td>[{"id": 246, "name": "dancing"}, {"id": 271, "...</td>
      <td>cs</td>
      <td>Dancin' It's On</td>
      <td>This coming of age Dance Film, in the spirit o...</td>
      <td>0.186234</td>
      <td>[]</td>
      <td>...</td>
      <td>2015-10-16</td>
      <td>0</td>
      <td>89.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Post Production</td>
      <td>Dancin' Like You've Never Seen Before</td>
      <td>Dancin' It's On</td>
      <td>4.3</td>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4586</th>
      <td>35000000</td>
      <td>[{"id": 35, "name": "Comedy"}, {"id": 10749, "...</td>
      <td>NaN</td>
      <td>9029</td>
      <td>[{"id": 585, "name": "casino"}, {"id": 612, "n...</td>
      <td>da</td>
      <td>What Happens in Vegas</td>
      <td>During a wild vacation in Las Vegas, career wo...</td>
      <td>38.100488</td>
      <td>[{"name": "Twentieth Century Fox Film Corporat...</td>
      <td>...</td>
      <td>2008-05-07</td>
      <td>170000000</td>
      <td>99.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Get Lucky</td>
      <td>What Happens in Vegas</td>
      <td>5.8</td>
      <td>923</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




Python でデータ処理するライブラリの定番 Pandas の groupby がなかなか難しいので整理する。特に apply の仕様はパラメータの関数の戻り値によって予想外の振る舞いをするので凶悪に思える。

まず必要なライブラリを import する。


```python
import pandas as pd
import numpy as np
```

## DataFrame を作る

サンプル用のデータを適当に作る。


```python
df = pd.DataFrame({
    'city': ['osaka', 'osaka', 'osaka', 'osaka', 'tokyo', 'tokyo', 'tokyo'],
    'food': ['apple', 'orange', 'banana', 'banana', 'apple', 'apple', 'banana'],
    'price': [100, 200, 250, 300, 150, 200, 400],
    'quantity': [1, 2, 3, 4, 5, 6, 7]
})
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>food</th>
      <th>price</th>
      <th>quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>osaka</td>
      <td>apple</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>osaka</td>
      <td>orange</td>
      <td>200</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>osaka</td>
      <td>banana</td>
      <td>250</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>osaka</td>
      <td>banana</td>
      <td>300</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tokyo</td>
      <td>apple</td>
      <td>150</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tokyo</td>
      <td>apple</td>
      <td>200</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>tokyo</td>
      <td>banana</td>
      <td>400</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



余談だが、本題に入る前に Pandas の二次元データ構造 DataFrame について軽く触れる。余談だが Pandas は列志向のデータ構造なので、データの作成は縦にカラムごとに行う。列ごとの処理は得意で速いが、行ごとの処理はイテレータ等を使って Python の世界で行うので遅くなる。

DataFrame には index と呼ばれる特殊なリストがある。上の例では、`'city', 'food', 'price'` のように各列を表す index と `0, 1, 2, 3, ...` のように各行を表す index がある。また、各 index の要素を label と呼ぶ。それぞれ以下のようなプロパティで取得出来る。


```python
df.columns # 列 label の取得
```




    Index(['city', 'food', 'price', 'quantity'], dtype='object')




```python
df.index # 行 label の取得
```




    RangeIndex(start=0, stop=7, step=1)



## 簡単な groupby の使い方

余談終わり。groupby は、同じ値を持つデータをまとめて、それぞれの塊に対して共通の操作を行いたい時に使う。例えば一番簡単な使い方として、city ごとの price の平均を求めるには次のようにする。groupby で出来た [GroupBy](https://pandas.pydata.org/pandas-docs/stable/api.html#groupby) オブジェクトに対して、平均をとる [mean](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.core.groupby.GroupBy.mean.html#pandas.core.groupby.GroupBy.mean) メソッドを呼ぶと良い。


```python
df.groupby('city').mean()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>quantity</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>osaka</th>
      <td>212.5</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>tokyo</th>
      <td>250.0</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>



グループの指定に複数の label を指定する事も出来る。city と food の組み合わせで平均をとるには次のようにする。


```python
df.groupby(['city', 'food']).mean()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>price</th>
      <th>quantity</th>
    </tr>
    <tr>
      <th>city</th>
      <th>food</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">osaka</th>
      <th>apple</th>
      <td>100.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>banana</th>
      <td>275.0</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>orange</th>
      <td>200.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">tokyo</th>
      <th>apple</th>
      <td>175.0</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>banana</th>
      <td>400.0</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>



groupby を使うと、デフォルトでグループラベルが index になる。index にしたく無い場合は `as_index=False` を指定する。


```python
df.groupby(['city', 'food'], as_index=False).mean()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>food</th>
      <th>price</th>
      <th>quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>osaka</td>
      <td>apple</td>
      <td>100.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>osaka</td>
      <td>banana</td>
      <td>275.0</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>osaka</td>
      <td>orange</td>
      <td>200.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tokyo</td>
      <td>apple</td>
      <td>175.0</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tokyo</td>
      <td>banana</td>
      <td>400.0</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>



## GroupBy オブジェクトの性質

デバッグ以外で使うところは無いかも知れないが、groupby によって作られた GroupBy オブジェクトの性質を調べるプロパティが幾つかある。まず、groupby によってどのように DataFrame が分割されたかを知るには groups を使う。`{ 列 label: [行 label, 行 label, ...], ... }` のような形で、どのグループにどの列が入ったか分かる。


```python
df.groupby('city').groups
```




    {'osaka': Int64Index([0, 1, 2, 3], dtype='int64'),
     'tokyo': Int64Index([4, 5, 6], dtype='int64')}



あるグループにどのようなデータが入ったかを知るには get_group を使う。


```python
df.groupby('city').get_group('osaka')
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>food</th>
      <th>price</th>
      <th>quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>osaka</td>
      <td>apple</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>osaka</td>
      <td>orange</td>
      <td>200</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>osaka</td>
      <td>banana</td>
      <td>250</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>osaka</td>
      <td>banana</td>
      <td>300</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



各グループのサイズは size で取得出来る。


```python
df.groupby('city').size()
```




    city
    osaka    4
    tokyo    3
    dtype: int64



size の結果は [Series](https://pandas.pydata.org/pandas-docs/stable/api.html#series) という一次元列を表すオブジェクトが返る。Series を使うと、osaka グループのサイズは添字を使って取得出来る。


```python
df.groupby('city').size()['osaka']
```




    4



## さまざまな Aggregation

GroupBy.mean() のように、グループごとに値を求めて表を作るような操作を Aggregation と呼ぶ。このように GroupBy オブジェクトには Aggregation に使う関数が幾つか定義されているが、これらは [agg()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.core.groupby.DataFrameGroupBy.agg.html#pandas.core.groupby.DataFrameGroupBy.agg) を使っても実装出来る。


```python
df.groupby('city').agg(np.mean)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>quantity</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>osaka</th>
      <td>212.5</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>tokyo</th>
      <td>250.0</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>



agg には多様な使い方がある。上の例では、mean() を使って各グループごとに price と quantity 両方の平均を求めたが、例えば price の平均と quantity の合計を同時に知りたいときは以下のように { グループ名: 関数 } の dict を渡す。関数には Series を受け取って一つの値を返す物を期待されている。 


```python
def my_mean(s):
    """わざとらしいサンプル"""
    return np.mean(s)

df.groupby('city').agg({'price': my_mean, 'quantity': np.sum})
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>quantity</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>osaka</th>
      <td>212.5</td>
      <td>10</td>
    </tr>
    <tr>
      <th>tokyo</th>
      <td>250.0</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>



## Group ごとに複数行を返す

Aggregation の結果はグループごとに一行にまとめられるが、もっと柔軟に結果を作りたいときは [apply](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.core.groupby.GroupBy.apply.html#pandas.core.groupby.GroupBy.apply) を使う。apply に渡す関数には get_group で得られるようなグループごとの DataFrame が渡される。グループ名は df.name で取得出来る。

### apply 関数の結果としてスカラを返す場合。全体の結果は Series になる。

* groupby で作った label が結果の row index になる。
* 行数はグループの数と同じになる。
* as_index の効果は無い。


```python
df.groupby(['city', 'food'], as_index=False).apply(lambda d: (d.price * d.quantity).sum())
```




    city   food  
    osaka  apple      100
           banana    1950
           orange     400
    tokyo  apple     1950
           banana    2800
    dtype: int64



グループ名 にアクセスしてみた例


```python
df.groupby(['city', 'food'], as_index=False).apply(lambda d: d.name)
```




    city   food  
    osaka  apple      (osaka, apple)
           banana    (osaka, banana)
           orange    (osaka, orange)
    tokyo  apple      (tokyo, apple)
           banana    (tokyo, banana)
    dtype: object



### apply 関数の結果として Series を返す場合。全体の結果は Series になる。

* groupby で作った label に加えて、apply 関数の結果の index が結果全体の row index になる。
* 全体の行数は関数から返す結果に依存する。
* as_index=False を指定すると、index が消えて連番になる。


```python
def total_series(d):
    return d.price * d.quantity

df.groupby(['city', 'food']).apply(total_series)
```




    city   food     
    osaka  apple   0     100
           banana  2     750
                   3    1200
           orange  1     400
    tokyo  apple   4     750
                   5    1200
           banana  6    2800
    dtype: int64



### apply 関数の結果として元の row index を保存した DataFrame を返す場合

DataFrame を返す場合、返す DataFrame に含まれる row index によって振る舞いが違う。非常に凶悪な仕様!!!!

元の index を保存した場合、下記 Transformation と同じ動作ように groupby の label は消える。

* apply 関数の結果を連結した DataFrame が作られる。groupby で対象になる label は index にならない。
* as_index=False の効果なし


```python
def total_keepindex(d):
    return pd.DataFrame({
        'total': d.price * d.quantity # ここで返る DataFrame の row index は d の row index と同じ
    })

df.groupby(['city', 'food']).apply(total_keepindex)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>400</td>
    </tr>
    <tr>
      <th>2</th>
      <td>750</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>750</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1200</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2800</td>
    </tr>
  </tbody>
</table>
</div>



### apply 関数の結果として元の row index を保存しない DataFrame を返す場合

元の index を保存しないと groupby で作った label が結果の row index になる。

* groupby で作った label が結果の row index になる。
* as_index=False の効果あり


```python
def total_keepnoindex(d):
    return pd.DataFrame({
        'total': (d.price * d.quantity).sum()
    }, index=['hoge'])
df.groupby(['city', 'food']).apply(total_keepnoindex)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>total</th>
    </tr>
    <tr>
      <th>city</th>
      <th>food</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">osaka</th>
      <th>apple</th>
      <th>hoge</th>
      <td>100</td>
    </tr>
    <tr>
      <th>banana</th>
      <th>hoge</th>
      <td>1950</td>
    </tr>
    <tr>
      <th>orange</th>
      <th>hoge</th>
      <td>400</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">tokyo</th>
      <th>apple</th>
      <th>hoge</th>
      <td>1950</td>
    </tr>
    <tr>
      <th>banana</th>
      <th>hoge</th>
      <td>2800</td>
    </tr>
  </tbody>
</table>
</div>



### 注意!! apply 関数が一度も呼ばれないとカラムが出来ない

Pandas の凶悪な所でありまた動的型付け言語の欠点なのだが、apply 関数の結果で動的にカラムを決めているからか、ゼロ行の DataFrame に対して apply を実行するとカラムが作成されない。ゼロ行だけ特別扱いしないと行けないので分かりづらいバグを生む。

例えばこの場合3行の DataFrame の場合カラムが出来る。


```python
pd.DataFrame({'hoge': [1,1,3], 'fuga': [10, 20, 30]}).groupby('hoge').apply(np.sum)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hoge</th>
      <th>fuga</th>
    </tr>
    <tr>
      <th>hoge</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>



ところがゼロ行の DataFrame に対して同じ apply を実行するとカラムが消えてしまう。


```python
pd.DataFrame({'hoge': [], 'fuga': []}).groupby('hoge').apply(np.sum)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
    </tr>
    <tr>
      <th>hoge</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



## Transformation

グループごとの統計情報を使ってすべての行を集計したい場合は Transformation を使う。説明が難しい。。。transformation の引数にはグループごとの列の Series が与えられる。戻り値は引数と同様の Series かスカラを渡す。スカラを渡した場合は引数と同じ個数だけ繰り返される。

例えば、グループごとに各アイテムの割合を求めるには次のようにする。


```python
def transformation_sample(s):
    return (s / s.sum() * 100).astype(str) + '%'

df.groupby(['city']).transform(transformation_sample)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11.76470588235294%</td>
      <td>10.0%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>23.52941176470588%</td>
      <td>20.0%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29.411764705882355%</td>
      <td>30.0%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.294117647058826%</td>
      <td>40.0%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.0%</td>
      <td>27.77777777777778%</td>
    </tr>
    <tr>
      <th>5</th>
      <td>26.666666666666668%</td>
      <td>33.33333333333333%</td>
    </tr>
    <tr>
      <th>6</th>
      <td>53.333333333333336%</td>
      <td>38.88888888888889%</td>
    </tr>
  </tbody>
</table>
</div>



## 参考

* ガイド: [Group By: split-apply-combine](https://pandas.pydata.org/pandas-docs/stable/groupby.html)
* groupby 関数のリファレンス: [pandas.DataFrame.groupby](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html)
* GroupBy オブジェクトのリファレンス: [GroupBy](https://pandas.pydata.org/pandas-docs/stable/api.html#groupby)
* [Series](https://pandas.pydata.org/pandas-docs/stable/api.html#series)
* [DataFrame](https://pandas.pydata.org/pandas-docs/stable/api.html#dataframe)

